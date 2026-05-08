"""CLI rationale-rerank: kill-product test, no embedding.

For each query: generate one-paragraph bridge rationales per candidate (parallel),
then a single LLM rerank call over query + rationale-augmented candidates.

No Voyage / API embeddings — uses lightweight BM25-style word-overlap to cap the
candidate pool when --candidate-pool > 0. At --candidate-pool 0 (default) the
rationale step runs against the full corpus.
"""
import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.cli_invoker import cli_invoke, cli_invoke_many, parse_json_response  # noqa: E402

METHOD_NAME = "cli-rationale-rerank"
TOKEN_RE = re.compile(r"[a-z0-9]+")


def file_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def env_summary() -> dict:
    return {"python": sys.version.split()[0]}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def word_overlap_score(query_tokens: set[str], note: dict) -> int:
    note_tokens = set(tokenize(note["title"] + " " + note["body"]))
    return len(query_tokens & note_tokens)


def select_candidates(query: str, notes: list[dict], cap: int) -> list[dict]:
    if cap <= 0 or cap >= len(notes):
        return notes
    qtok = set(tokenize(query))
    scored = [(n, word_overlap_score(qtok, n)) for n in notes]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [n for n, _ in scored[:cap]]


def build_rationale_prompt(query: str, note: dict) -> str:
    return f"""How does this note connect to the query? What's the cross-domain insight or conceptual bridge?

Query: {query}

Note title: {note['title']}
Note body: {note['body'][:1500]}

Write a one-paragraph rationale (3-5 sentences) explaining the connection. Be concrete about the bridge — name the underlying structure both share."""


def build_final_rerank_prompt(query: str, augmented: list[dict], top_k: int) -> str:
    block = "\n\n".join(
        f"id: {a['id']}\ntitle: {a['title']}\nbody: {a['body'][:600]}\n\nBridge rationale: {a['rationale']}"
        for a in augmented
    )
    return f"""Rank these candidate notes by relevance to the query. Each candidate has been augmented with a bridge rationale describing how it might connect to the query — use both the note content AND the rationale when judging.

The query is looking for cross-domain conceptual bridges — notes that share underlying structural ideas with the question, even when surface vocabulary differs.

Query: {query}

Candidates:
{block}

Respond with ONLY a JSON array of the top {top_k} note IDs in order of relevance, e.g. ["ai-02", "qf-01", "eq-04", "ph-03", "sa-05"]. No prose, no code fence."""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--candidate-pool", type=int, default=0,
                    help="Cap rationale candidates to top-N by word-overlap. 0 = full corpus.")
    ap.add_argument("--cli", choices=["claude", "codex", "gemini"], default="claude")
    ap.add_argument("--model", default=None)
    ap.add_argument("--concurrency", type=int, default=1,
                    help="Parallel rationale-generation calls per query. Default 1 (sequential) — restart-safe.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip query_ids already present in --out; append rather than overwrite.")
    args = ap.parse_args()

    notes = [json.loads(line) for line in args.corpus.read_text().splitlines() if line.strip()]
    note_ids = {n["id"] for n in notes}
    queries = [json.loads(line) for line in args.queries.read_text().splitlines() if line.strip()]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    traces_dir = args.out.parent / "traces" / args.out.stem
    traces_dir.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if args.resume and args.out.exists():
        for line in args.out.read_text().splitlines():
            if not line.strip():
                continue
            try:
                done.add(json.loads(line)["query_id"])
            except (json.JSONDecodeError, KeyError):
                continue
        print(f"resume: {len(done)} query_ids already done; skipping", flush=True)

    started = time.time()
    open_mode = "a" if args.resume and args.out.exists() else "w"
    with args.out.open(open_mode, buffering=1) as f:  # line-buffered, survives subprocess death
        for q in queries:
            if q["query_id"] in done:
                continue
            q_started = time.time()
            candidates = select_candidates(q["query_text"], notes, args.candidate_pool)

            # Step 1: parallel rationale generation
            items = [(q["query_text"], n) for n in candidates]
            def invoker(item):
                qt, n = item
                return cli_invoke(args.cli, build_rationale_prompt(qt, n), model=args.model, timeout=240)
            responses = cli_invoke_many(items, invoker, max_concurrency=args.concurrency)
            rationales = [r.text for r in responses]
            rationale_traces = [{"note_id": n["id"], "wall_clock_seconds": r.wall_clock_seconds,
                                  "prompt": build_rationale_prompt(q["query_text"], n), "rationale": r.text}
                                 for n, r in zip(candidates, responses)]
            augmented = [{"id": n["id"], "title": n["title"], "body": n["body"], "rationale": r}
                         for n, r in zip(candidates, rationales)]

            # Step 2: final rerank
            rerank_prompt = build_final_rerank_prompt(q["query_text"], augmented, args.top_k)
            rr = cli_invoke(args.cli, rerank_prompt, model=args.model, timeout=300)
            parsed = parse_json_response(rr.text)
            parse_error = None
            top_ids: list[str] = []
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, str) and item in note_ids and item not in top_ids:
                        top_ids.append(item)
                    if len(top_ids) >= args.top_k:
                        break
                if len(top_ids) < args.top_k:
                    parse_error = f"received {len(parsed)} items, only {len(top_ids)} valid corpus IDs"
            else:
                parse_error = f"failed to parse list of strings from {rr.text[:200]!r}"

            # Pad fallback: candidates by word-overlap order
            if len(top_ids) < args.top_k:
                for n in candidates:
                    if n["id"] not in top_ids:
                        top_ids.append(n["id"])
                    if len(top_ids) >= args.top_k:
                        break

            top_k_records = [{"id": nid, "title": next(n["title"] for n in notes if n["id"] == nid)} for nid in top_ids]
            f.write(json.dumps({
                "query_id": q["query_id"],
                "query_text": q["query_text"],
                "top_k": top_k_records,
                "method": METHOD_NAME,
            }) + "\n")
            f.flush()  # belt-and-braces — survives subprocess death
            (traces_dir / f"{q['query_id']}.json").write_text(json.dumps({
                "query_id": q["query_id"],
                "query_text": q["query_text"],
                "candidate_pool_size": len(candidates),
                "candidates": [{"id": n["id"]} for n in candidates],
                "rationales": rationale_traces,
                "rerank_prompt": rerank_prompt,
                "rerank_raw_response": rr.text,
                "rerank_parsed": parsed,
                "rerank_parse_error": parse_error,
                "rerank_wall_clock_seconds": rr.wall_clock_seconds,
                "final_top_k": top_k_records,
                "query_wall_clock_seconds": round(time.time() - q_started, 2),
            }, indent=2))
            print(f"  {q['query_id']}: {len(candidates)} rationales + 1 rerank, {round(time.time() - q_started, 1)}s, top1={top_ids[0]}", flush=True)

    manifest = {
        "method": METHOD_NAME,
        "started_at": datetime.fromtimestamp(started, tz=timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "wall_clock_seconds": round(time.time() - started, 2),
        "cli": args.cli,
        "model_override": args.model,
        "concurrency": args.concurrency,
        "candidate_pool_cap": args.candidate_pool,
        "candidate_selection": "word-overlap (no embedding)" if args.candidate_pool > 0 else "full corpus",
        "temperature": "cli-default",
        "corpus_sha256": file_checksum(args.corpus),
        "corpus_count": len(notes),
        "queries_sha256": file_checksum(args.queries),
        "queries_count": len(queries),
        "top_k": args.top_k,
        "env": env_summary(),
    }
    (traces_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(queries)} results to {args.out}; traces in {traces_dir}; total {manifest['wall_clock_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
