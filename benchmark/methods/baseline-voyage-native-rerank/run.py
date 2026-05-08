"""Voyage-3 + Voyage native reranker baseline (no LLM in the loop).

Embed corpus + queries with Voyage-3, cosine-rank to full corpus (v0.1 scale),
rerank via Voyage's rerank-2, take top-5. Full traces written for reproducibility.
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
EMBED_MODEL = "voyage-3"
RERANK_MODEL = "rerank-2"
METHOD_NAME = "voyage-native-rerank"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def file_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def embed_corpus(notes: list[dict], vclient: voyageai.Client) -> dict[str, np.ndarray]:
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / f"corpus-{EMBED_MODEL}.npz"
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        cached = {str(k): data[k] for k in data.files}
        if all(n["id"] in cached for n in notes):
            return cached
    texts = [f"{n['title']}\n\n{n['body']}" for n in notes]
    result = vclient.embed(texts, model=EMBED_MODEL, input_type="document")
    by_id = {n["id"]: np.array(emb) for n, emb in zip(notes, result.embeddings)}
    np.savez(cache_path, **{k: v for k, v in by_id.items()})
    return by_id


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    notes = [json.loads(line) for line in args.corpus.read_text().splitlines() if line.strip()]
    queries = [json.loads(line) for line in args.queries.read_text().splitlines() if line.strip()]

    vclient = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    note_embeddings = embed_corpus(notes, vclient)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    traces_dir = args.out.parent / "traces" / args.out.stem
    traces_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    with args.out.open("w") as f:
        for q in queries:
            q_emb = np.array(vclient.embed([q["query_text"]], model=EMBED_MODEL, input_type="query").embeddings[0])
            scored = [(n, cosine(q_emb, note_embeddings[n["id"]])) for n in notes]
            candidates = sorted(scored, key=lambda x: x[1], reverse=True)
            candidate_notes = [n for n, _ in candidates]
            candidate_texts = [f"{n['title']}\n\n{n['body']}" for n in candidate_notes]

            rr_result = vclient.rerank(query=q["query_text"], documents=candidate_texts, model=RERANK_MODEL, top_k=args.top_k)
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
                "embed_model": EMBED_MODEL,
                "rerank_model": RERANK_MODEL,
                "candidate_pool_size": len(candidate_notes),
                "cosine_top_5_pre_rerank": [{"id": n["id"], "score": s} for n, s in candidates[:5]],
                "reranked_top_k": [{"id": n["id"], "title": n["title"], "score": s} for n, s in zip(reranked, scores)],
            }, indent=2))

    manifest = {
        "method": METHOD_NAME,
        "started_at": datetime.fromtimestamp(started, tz=timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "wall_clock_seconds": round(time.time() - started, 2),
        "embed_model": EMBED_MODEL,
        "rerank_model": RERANK_MODEL,
        "corpus_checksum": file_checksum(args.corpus),
        "corpus_count": len(notes),
        "queries_checksum": file_checksum(args.queries),
        "queries_count": len(queries),
        "candidate_pool_size": len(notes),
        "top_k": args.top_k,
    }
    (traces_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(queries)} results to {args.out}; traces in {traces_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
