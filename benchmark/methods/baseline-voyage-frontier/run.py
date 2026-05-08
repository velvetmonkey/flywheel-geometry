"""Frontier Voyage baseline (voyage-4-large + rerank-2.5).

Mirrors baseline-voyage-native-rerank but with current-frontier model strings
configurable via --embed-model and --rerank-model. Results from this method
should be treated as a moving target, not a stable v0.1 comparison anchor.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import voyageai

CACHE_DIR = Path(__file__).parent / ".cache"
METHOD_NAME = "voyage-frontier"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def file_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def env_summary() -> dict:
    info = {"python": sys.version.split()[0]}
    try:
        from importlib.metadata import version
        for pkg in ["voyageai", "numpy"]:
            try:
                info[pkg] = version(pkg)
            except Exception:
                info[pkg] = None
    except Exception:
        pass
    return info


def embed_corpus(notes: list[dict], vclient: voyageai.Client, corpus_path: Path, embed_model: str) -> dict[str, np.ndarray]:
    CACHE_DIR.mkdir(exist_ok=True)
    sha = file_checksum(corpus_path)[:16]
    cache_path = CACHE_DIR / f"corpus-{embed_model}-{sha}.npz"
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return {str(k): data[k] for k in data.files}
    texts = [f"{n['title']}\n\n{n['body']}" for n in notes]
    result = vclient.embed(texts, model=embed_model, input_type="document")
    by_id = {n["id"]: np.array(emb) for n, emb in zip(notes, result.embeddings)}
    np.savez(cache_path, **{k: v for k, v in by_id.items()})
    return by_id


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--embed-model", default="voyage-4-large")
    ap.add_argument("--rerank-model", default="rerank-2.5")
    args = ap.parse_args()

    notes = [json.loads(line) for line in args.corpus.read_text().splitlines() if line.strip()]
    queries = [json.loads(line) for line in args.queries.read_text().splitlines() if line.strip()]

    vclient = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    note_embeddings = embed_corpus(notes, vclient, args.corpus, args.embed_model)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    traces_dir = args.out.parent / "traces" / args.out.stem
    traces_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    with args.out.open("w") as f:
        for q in queries:
            q_emb = np.array(vclient.embed([q["query_text"]], model=args.embed_model, input_type="query").embeddings[0])
            scored = [(n, cosine(q_emb, note_embeddings[n["id"]])) for n in notes]
            sorted_full = sorted(scored, key=lambda x: x[1], reverse=True)
            candidate_notes = [n for n, _ in sorted_full]
            candidate_texts = [f"{n['title']}\n\n{n['body']}" for n in candidate_notes]

            rr_result = vclient.rerank(query=q["query_text"], documents=candidate_texts, model=args.rerank_model, top_k=args.top_k)
            reranked = [candidate_notes[r.index] for r in rr_result.results]
            scores = [r.relevance_score for r in rr_result.results]

            f.write(json.dumps({
                "query_id": q["query_id"],
                "query_text": q["query_text"],
                "top_k": [{"id": n["id"], "title": n["title"], "score": s} for n, s in zip(reranked, scores)],
                "method": METHOD_NAME,
            }) + "\n")
            (traces_dir / f"{q['query_id']}.json").write_text(json.dumps({
                "query_id": q["query_id"],
                "query_text": q["query_text"],
                "embed_model": args.embed_model,
                "rerank_model": args.rerank_model,
                "candidate_pool_size": len(candidate_notes),
                "cosine_top_5_pre_rerank": [{"id": n["id"], "score": s} for n, s in sorted_full[:5]],
                "reranked_top_k": [{"id": n["id"], "title": n["title"], "score": s} for n, s in zip(reranked, scores)],
            }, indent=2))

    manifest = {
        "method": METHOD_NAME,
        "started_at": datetime.fromtimestamp(started, tz=timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "wall_clock_seconds": round(time.time() - started, 2),
        "embed_model": args.embed_model,
        "rerank_model": args.rerank_model,
        "corpus_sha256": file_checksum(args.corpus),
        "corpus_count": len(notes),
        "queries_sha256": file_checksum(args.queries),
        "queries_count": len(queries),
        "candidate_pool_size": len(notes),
        "top_k": args.top_k,
        "env": env_summary(),
        "note": "Frontier baseline — model strings are moving targets; not a stable v0.1 comparison anchor.",
    }
    (traces_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(queries)} results to {args.out}; traces in {traces_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
