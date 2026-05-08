"""Voyage-3 + Claude rerank WITH bridge-rationale generation (Codex #6 / kill-product test).

Full corpus reranking at v0.1 scale. For each candidate, generate a bridge rationale via Claude;
append rationale to candidate text; re-embed augmented text; re-rank by cosine to query.
Full traces written for reproducibility.
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
from anthropic import Anthropic

CACHE_DIR = Path(__file__).parent / ".cache"
RATIONALE_MODEL = "claude-sonnet-4-6"
EMBED_MODEL = "voyage-3"
METHOD_NAME = "voyage-rationale-rerank"


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


def generate_rationale(query: str, note: dict, aclient: Anthropic) -> tuple[str, dict]:
    prompt = f"""How does this note connect to the query? What's the cross-domain insight or conceptual bridge?

Query: {query}

Note title: {note['title']}
Note body: {note['body'][:1500]}

Write a one-paragraph rationale (3-5 sentences) explaining the connection. Be concrete about the bridge — name the underlying structure both share."""
    msg = aclient.messages.create(
        model=RATIONALE_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    rationale = msg.content[0].text.strip()
    return rationale, {"rationale_model": RATIONALE_MODEL, "prompt": prompt, "rationale": rationale}


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
    aclient = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    note_embeddings = embed_corpus(notes, vclient)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    traces_dir = args.out.parent / "traces" / args.out.stem
    traces_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    with args.out.open("w") as f:
        for q in queries:
            q_emb = np.array(vclient.embed([q["query_text"]], model=EMBED_MODEL, input_type="query").embeddings[0])
            scored = [(n, cosine(q_emb, note_embeddings[n["id"]])) for n in notes]
            candidates_full = sorted(scored, key=lambda x: x[1], reverse=True)
            candidate_notes = [n for n, _ in candidates_full]

            rationale_traces = []
            rationales = []
            for n in candidate_notes:
                rationale, t = generate_rationale(q["query_text"], n, aclient)
                rationales.append(rationale)
                rationale_traces.append({"note_id": n["id"], **t})

            augmented_texts = [f"{n['title']}\n\n{n['body']}\n\nBridge rationale: {r}" for n, r in zip(candidate_notes, rationales)]
            aug_embs = [np.array(e) for e in vclient.embed(augmented_texts, model=EMBED_MODEL, input_type="document").embeddings]
            rescored = [(n, r, cosine(q_emb, e)) for n, r, e in zip(candidate_notes, rationales, aug_embs)]
            reranked = sorted(rescored, key=lambda x: x[2], reverse=True)[: args.top_k]

            f.write(json.dumps({
                "query_id": q["query_id"],
                "query_text": q["query_text"],
                "top_k": [{"id": n["id"], "title": n["title"], "rescore": s} for n, _, s in reranked],
                "method": METHOD_NAME,
            }) + "\n")
            (traces_dir / f"{q['query_id']}.json").write_text(json.dumps({
                "query_id": q["query_id"],
                "query_text": q["query_text"],
                "embed_model": EMBED_MODEL,
                "rationale_model": RATIONALE_MODEL,
                "candidate_pool_size": len(candidate_notes),
                "cosine_top_5_pre_rationale": [{"id": n["id"], "score": s} for n, s in candidates_full[:5]],
                "rationales": rationale_traces,
                "rescored_top_k": [{"id": n["id"], "rationale": r, "rescore": s} for n, r, s in reranked],
            }, indent=2))

    manifest = {
        "method": METHOD_NAME,
        "started_at": datetime.fromtimestamp(started, tz=timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "wall_clock_seconds": round(time.time() - started, 2),
        "embed_model": EMBED_MODEL,
        "rationale_model": RATIONALE_MODEL,
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
