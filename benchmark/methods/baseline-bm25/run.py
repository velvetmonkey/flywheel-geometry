"""BM25 baseline runner.

Reads corpus + queries, outputs top-5 retrieved IDs per query as JSONL.
"""
import argparse
import json
import sys
from pathlib import Path

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    notes = [json.loads(line) for line in args.corpus.read_text().splitlines() if line.strip()]
    queries = [json.loads(line) for line in args.queries.read_text().splitlines() if line.strip()]

    tokenized_corpus = [tokenize(f"{n['title']} {n['body']}") for n in notes]
    bm25 = BM25Okapi(tokenized_corpus)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for q in queries:
            scores = bm25.get_scores(tokenize(q["query_text"]))
            ranked = sorted(zip(notes, scores), key=lambda x: x[1], reverse=True)[: args.top_k]
            f.write(json.dumps({
                "query_id": q["query_id"],
                "query_text": q["query_text"],
                "top_k": [{"id": n["id"], "title": n["title"], "score": float(s)} for n, s in ranked],
                "method": "bm25",
            }) + "\n")
    print(f"wrote {len(queries)} results to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
