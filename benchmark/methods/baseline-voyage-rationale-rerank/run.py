"""Voyage-3 + LLM rerank WITH bridge-rationale generation (Codex #6 / kill-product test).

For each top-15 candidate, generate a bridge rationale via Claude; append to candidate text;
re-embed with rationale; re-rank by cosine to query.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import voyageai
from anthropic import Anthropic

CACHE_DIR = Path(__file__).parent / ".cache"
RATIONALE_MODEL = "claude-sonnet-4-6"
EMBED_MODEL = "voyage-3"
TOP_CANDIDATES = 15


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


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


def generate_rationale(query: str, note: dict, aclient: Anthropic) -> str:
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
    return msg.content[0].text.strip()


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

    with args.out.open("w") as f:
        for q in queries:
            q_emb = np.array(vclient.embed([q["query_text"]], model=EMBED_MODEL, input_type="query").embeddings[0])
            scored = [(n, cosine(q_emb, note_embeddings[n["id"]])) for n in notes]
            top_candidates = [n for n, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:TOP_CANDIDATES]]

            rationales = [generate_rationale(q["query_text"], n, aclient) for n in top_candidates]
            augmented_texts = [f"{n['title']}\n\n{n['body']}\n\nBridge rationale: {r}" for n, r in zip(top_candidates, rationales)]
            aug_embs = [np.array(e) for e in vclient.embed(augmented_texts, model=EMBED_MODEL, input_type="document").embeddings]
            rescored = [(n, r, cosine(q_emb, e)) for n, r, e in zip(top_candidates, rationales, aug_embs)]
            reranked = sorted(rescored, key=lambda x: x[2], reverse=True)[: args.top_k]

            f.write(json.dumps({
                "query_id": q["query_id"],
                "query_text": q["query_text"],
                "top_k": [{"id": n["id"], "title": n["title"], "rationale": r, "rescore": s} for n, r, s in reranked],
                "method": "voyage-3+rationale+rerank",
            }) + "\n")
    print(f"wrote {len(queries)} results to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
