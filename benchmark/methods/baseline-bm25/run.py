"""BM25 baseline runner.

Reads corpus + queries, outputs top-5 retrieved IDs per query as JSONL.
Writes traces + manifest for reproducibility.
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from rank_bm25 import BM25Okapi

METHOD_NAME = "bm25"


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def file_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def env_summary() -> dict:
    info = {"python": sys.version.split()[0]}
    try:
        from importlib.metadata import version
        for pkg in ["rank_bm25", "numpy"]:
            try:
                info[pkg] = version(pkg)
            except Exception:
                info[pkg] = None
    except Exception:
        pass
    return info


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
    traces_dir = args.out.parent / "traces" / args.out.stem
    traces_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    with args.out.open("w") as f:
        for q in queries:
            tokens = tokenize(q["query_text"])
            scores = bm25.get_scores(tokens)
            ranked = sorted(zip(notes, scores), key=lambda x: x[1], reverse=True)
            top_k = ranked[: args.top_k]
            f.write(json.dumps({
                "query_id": q["query_id"],
                "query_text": q["query_text"],
                "top_k": [{"id": n["id"], "title": n["title"], "score": float(s)} for n, s in top_k],
                "method": METHOD_NAME,
            }) + "\n")
            (traces_dir / f"{q['query_id']}.json").write_text(json.dumps({
                "query_id": q["query_id"],
                "query_text": q["query_text"],
                "tokens": tokens,
                "all_scores": [{"id": n["id"], "score": float(s)} for n, s in ranked],
                "top_k": [{"id": n["id"], "title": n["title"], "score": float(s)} for n, s in top_k],
            }, indent=2))

    manifest = {
        "method": METHOD_NAME,
        "started_at": datetime.fromtimestamp(started, tz=timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "wall_clock_seconds": round(time.time() - started, 2),
        "tokenizer": "lowercase-whitespace",
        "bm25_params": {"k1": 1.5, "b": 0.75},
        "corpus_sha256": file_checksum(args.corpus),
        "corpus_count": len(notes),
        "queries_sha256": file_checksum(args.queries),
        "queries_count": len(queries),
        "top_k": args.top_k,
        "env": env_summary(),
    }
    (traces_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(queries)} results to {args.out}; traces in {traces_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
