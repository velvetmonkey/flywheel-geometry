"""Voyage-3 + rationale-augmented + Claude/CLI final rerank (Method 4b).

Generates one-paragraph bridge rationales per candidate (parallel), then sends
query + rationale-augmented candidates to a single LLM rerank call that picks
the top-K indices. Distinct from Method 4a, which does the final rank with a
cosine pass over re-embedded augmented text.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.cli_invoker import cli_invoke, cli_invoke_many, parse_json_response  # noqa: E402

CACHE_DIR = Path(__file__).parent / ".cache"
EMBED_MODEL = "voyage-3"
METHOD_NAME = "voyage-rationale-claude-rerank"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def file_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def embed_corpus(notes: list[dict], vclient: voyageai.Client, corpus_path: Path) -> dict[str, np.ndarray]:
    CACHE_DIR.mkdir(exist_ok=True)
    sha = file_checksum(corpus_path)[:16]
    cache_path = CACHE_DIR / f"corpus-{EMBED_MODEL}-{sha}.npz"
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return {str(k): data[k] for k in data.files}
    texts = [f"{n['title']}\n\n{n['body']}" for n in notes]
    result = vclient.embed(texts, model=EMBED_MODEL, input_type="document")
    by_id = {n["id"]: np.array(emb) for n, emb in zip(notes, result.embeddings)}
    np.savez(cache_path, **{k: v for k, v in by_id.items()})
    return by_id


def build_rationale_prompt(query: str, note: dict) -> str:
    return f"""How does this note connect to the query? What's the cross-domain insight or conceptual bridge?

Query: {query}

Note title: {note['title']}
Note body: {note['body'][:1500]}

Write a one-paragraph rationale (3-5 sentences) explaining the connection. Be concrete about the bridge — name the underlying structure both share."""


def build_final_rerank_prompt(query: str, augmented: list[dict], top_k: int) -> str:
    block = "\n\n".join(
        f"[{i}] {a['title']}\n{a['body'][:600]}\n\nBridge rationale: {a['rationale']}"
        for i, a in enumerate(augmented)
    )
    return f"""Rank these candidate notes by relevance to the query. Each candidate has been augmented with a bridge rationale describing how it might connect to the query — use both the note content AND the rationale when judging.

Query: {query}

Candidates:
{block}

Respond with ONLY a JSON array of the top {top_k} candidate indices in order of relevance. Like [3, 0, 7, 1, 12]. No prose, no code fence."""


def env_summary() -> dict:
    """Capture Python + key dep versions for the manifest."""
    info = {"python": sys.version.split()[0]}
    try:
        from importlib.metadata import version
        for pkg in ["voyageai", "anthropic", "numpy"]:
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
    ap.add_argument("--candidate-pool", type=int, default=0,
                    help="Cap rationale candidates to top-N cosine. 0 = full corpus.")
    ap.add_argument("--llm", choices=["cli", "api"], default="cli")
    ap.add_argument("--cli", choices=["claude", "codex", "gemini"], default="claude")
    ap.add_argument("--model", default=None)
    ap.add_argument("--concurrency", type=int, default=5)
    args = ap.parse_args()

    notes = [json.loads(line) for line in args.corpus.read_text().splitlines() if line.strip()]
    queries = [json.loads(line) for line in args.queries.read_text().splitlines() if line.strip()]

    vclient = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    note_embeddings = embed_corpus(notes, vclient, args.corpus)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    traces_dir = args.out.parent / "traces" / args.out.stem
    traces_dir.mkdir(parents=True, exist_ok=True)

    api_model = args.model or "claude-sonnet-4-20250514"
    started = time.time()

    with args.out.open("w") as f:
        for q in queries:
            q_started = time.time()
            q_emb = np.array(vclient.embed([q["query_text"]], model=EMBED_MODEL, input_type="query").embeddings[0])
            scored = [(n, cosine(q_emb, note_embeddings[n["id"]])) for n in notes]
            sorted_full = sorted(scored, key=lambda x: x[1], reverse=True)
            cap = args.candidate_pool if args.candidate_pool > 0 else len(notes)
            candidate_notes = [n for n, _ in sorted_full[:cap]]

            # Step 1: rationales (parallel)
            if args.llm == "cli":
                items = [(q["query_text"], n) for n in candidate_notes]
                def invoker(item):
                    return cli_invoke(args.cli, build_rationale_prompt(*item), model=args.model, timeout=240)
                responses = cli_invoke_many(items, invoker, max_concurrency=args.concurrency)
                rationales = [r.text for r in responses]
                rationale_traces = [{"note_id": n["id"], "cli": r.cli, "model": r.model,
                                     "wall_clock_seconds": r.wall_clock_seconds}
                                    for n, r in zip(candidate_notes, responses)]
            else:
                from anthropic import Anthropic
                aclient = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                rationales = []
                rationale_traces = []
                for n in candidate_notes:
                    msg = aclient.messages.create(
                        model=api_model, max_tokens=300, temperature=0,
                        messages=[{"role": "user", "content": build_rationale_prompt(q["query_text"], n)}],
                    )
                    rationales.append(msg.content[0].text.strip())
                    rationale_traces.append({"note_id": n["id"], "model": api_model, "temperature": 0})

            augmented = [{"title": n["title"], "body": n["body"], "rationale": r} for n, r in zip(candidate_notes, rationales)]

            # Step 2: single LLM rerank over augmented candidates
            rerank_prompt = build_final_rerank_prompt(q["query_text"], augmented, args.top_k)
            if args.llm == "cli":
                resp = cli_invoke(args.cli, rerank_prompt, model=args.model, timeout=240)
                raw = resp.text
                rerank_meta = {"cli": resp.cli, "model": resp.model, "wall_clock_seconds": resp.wall_clock_seconds}
            else:
                from anthropic import Anthropic
                aclient = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                msg = aclient.messages.create(
                    model=api_model, max_tokens=200, temperature=0,
                    messages=[{"role": "user", "content": rerank_prompt}],
                )
                raw = msg.content[0].text.strip()
                rerank_meta = {"model": api_model, "temperature": 0}

            parsed = parse_json_response(raw)
            parse_error = None
            if isinstance(parsed, list) and all(isinstance(i, int) for i in parsed):
                try:
                    reranked = [candidate_notes[i] for i in parsed[:args.top_k]]
                except IndexError as e:
                    parse_error = f"index out of range: {e}"
                    reranked = candidate_notes[:args.top_k]
            else:
                parse_error = f"failed to parse list of ints from {raw[:200]!r}"
                reranked = candidate_notes[:args.top_k]

            f.write(json.dumps({
                "query_id": q["query_id"],
                "query_text": q["query_text"],
                "top_k": [{"id": n["id"], "title": n["title"]} for n in reranked],
                "method": METHOD_NAME,
            }) + "\n")
            (traces_dir / f"{q['query_id']}.json").write_text(json.dumps({
                "query_id": q["query_id"],
                "query_text": q["query_text"],
                "embed_model": EMBED_MODEL,
                "candidate_pool_size": len(candidate_notes),
                "cosine_top_5_pre_rationale": [{"id": n["id"], "score": s} for n, s in sorted_full[:5]],
                "rationales": [{"note_id": n["id"], "rationale": r} for n, r in zip(candidate_notes, rationales)],
                "rationale_traces": rationale_traces,
                "rerank_prompt": rerank_prompt,
                "rerank_raw_response": raw,
                "rerank_parsed": parsed,
                "rerank_parse_error": parse_error,
                "rerank_meta": rerank_meta,
                "final_top_k": [{"id": n["id"], "title": n["title"]} for n in reranked],
                "query_wall_clock_seconds": round(time.time() - q_started, 2),
            }, indent=2))
            print(f"  {q['query_id']}: {len(candidate_notes)} rationales + 1 rerank, {round(time.time() - q_started, 1)}s, top1={reranked[0]['id']}", flush=True)

    manifest = {
        "method": METHOD_NAME,
        "started_at": datetime.fromtimestamp(started, tz=timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "wall_clock_seconds": round(time.time() - started, 2),
        "embed_model": EMBED_MODEL,
        "llm_mode": args.llm,
        "cli": args.cli if args.llm == "cli" else None,
        "api_model": api_model if args.llm == "api" else None,
        "model_override": args.model,
        "concurrency": args.concurrency if args.llm == "cli" else 1,
        "temperature": 0 if args.llm == "api" else "cli-default",
        "corpus_sha256": file_checksum(args.corpus),
        "corpus_count": len(notes),
        "queries_sha256": file_checksum(args.queries),
        "queries_count": len(queries),
        "candidate_pool_cap": args.candidate_pool,
        "top_k": args.top_k,
        "env": env_summary(),
    }
    (traces_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(queries)} results to {args.out}; traces in {traces_dir}; total {manifest['wall_clock_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
