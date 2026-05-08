"""Voyage-3 + LLM rerank baseline.

Embed corpus + queries with Voyage-3, cosine-rank to top-15, Claude rerank to top-5.
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
RERANK_MODEL = "claude-sonnet-4-6"
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


def rerank_with_llm(query: str, candidates: list[dict], aclient: Anthropic, top_k: int) -> list[dict]:
    candidate_block = "\n\n".join(
        f"[{i}] {c['title']}\n{c['body'][:800]}" for i, c in enumerate(candidates)
    )
    prompt = f"""Rank these candidate notes by relevance to the query below. The query is looking for cross-domain conceptual bridges — notes from different domains that share underlying structure.

Query: {query}

Candidates:
{candidate_block}

Respond with a JSON array of the top {top_k} candidate indices in order of relevance, like [3, 0, 7, 1, 12]. Only the JSON, no other text."""
    msg = aclient.messages.create(
        model=RERANK_MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.content[0].text.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:-1])
    indices = json.loads(text)
    return [candidates[i] for i in indices[:top_k]]


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
            reranked = rerank_with_llm(q["query_text"], top_candidates, aclient, args.top_k)
            f.write(json.dumps({
                "query_id": q["query_id"],
                "query_text": q["query_text"],
                "top_k": [{"id": n["id"], "title": n["title"]} for n in reranked],
                "method": "voyage-3+claude-rerank",
            }) + "\n")
    print(f"wrote {len(queries)} results to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
