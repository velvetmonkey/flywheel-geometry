"""CLI-direct rerank: pure-LLM baseline, no embedding step.

Each query sees the full corpus in a single CLI call. Asks the model for top-5
note IDs. Subscription-only — uses claude/codex/gemini CLI.
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.cli_invoker import cli_invoke, parse_json_response  # noqa: E402

METHOD_NAME = "cli-direct-rerank"


def file_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def env_summary() -> dict:
    info = {"python": sys.version.split()[0]}
    return info


def build_prompt(query: str, notes: list[dict], top_k: int) -> str:
    block = "\n\n".join(
        f"id: {n['id']}\ntitle: {n['title']}\nbody: {n['body'][:1200]}"
        for n in notes
    )
    return f"""You are searching a personal knowledge vault for cross-domain conceptual bridges. The query below is looking for notes that share underlying conceptual structure with the question — even if the surface vocabulary is different. Notes from different domains that share structural ideas (feedback loops, regime change, calibration, exploration vs exploitation, tail dependence, etc.) should rank higher than notes that just share keywords.

Query: {query}

Corpus (50 notes):

{block}

Return ONLY a JSON array of the top {top_k} note IDs in order of relevance, e.g. ["ai-02", "qf-01", "eq-04", "ph-03", "sa-05"]. No prose, no code fence, no other text."""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--cli", choices=["claude", "codex", "gemini"], default="claude")
    ap.add_argument("--model", default=None, help="Optional model override")
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
            prompt = build_prompt(q["query_text"], notes, args.top_k)
            resp = cli_invoke(args.cli, prompt, model=args.model, timeout=300)
            raw = resp.text
            parsed = parse_json_response(raw)
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
                parse_error = f"failed to parse list of strings from {raw[:200]!r}"

            # Pad with deterministic fallback if parsing failed: first N notes by ID
            if len(top_ids) < args.top_k:
                for n in notes:
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
                "cli": resp.cli,
                "model": resp.model,
                "prompt": prompt,
                "raw_response": raw,
                "parsed": parsed,
                "parse_error": parse_error,
                "final_top_k": top_k_records,
                "wall_clock_seconds": resp.wall_clock_seconds,
            }, indent=2))
            print(f"  {q['query_id']}: {round(time.time() - q_started, 1)}s, top1={top_ids[0]}", flush=True)

    manifest = {
        "method": METHOD_NAME,
        "started_at": datetime.fromtimestamp(started, tz=timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "wall_clock_seconds": round(time.time() - started, 2),
        "cli": args.cli,
        "model_override": args.model,
        "temperature": "cli-default",
        "corpus_sha256": file_checksum(args.corpus),
        "corpus_count": len(notes),
        "queries_sha256": file_checksum(args.queries),
        "queries_count": len(queries),
        "candidate_pool_size": len(notes),
        "top_k": args.top_k,
        "env": env_summary(),
    }
    (traces_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(queries)} results to {args.out}; traces in {traces_dir}; total {manifest['wall_clock_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
