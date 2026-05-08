"""Voyage-3 + LLM rerank baseline (LLM-as-reranker, no rationale generation).

Default mode: --llm cli (uses subscription tooling — claude / codex / gemini).
Optional: --llm api (uses Anthropic SDK + ANTHROPIC_API_KEY).

Embed corpus + queries with Voyage-3, cosine-rank to full corpus (v0.1 scale),
LLM rerank to top-5. Full traces written for reproducibility.
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

# Allow imports from the shared lib regardless of CWD
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.cli_invoker import cli_invoke, parse_json_response  # noqa: E402

CACHE_DIR = Path(__file__).parent / ".cache"
EMBED_MODEL = "voyage-3"
METHOD_NAME = "voyage-rerank"


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


def build_rerank_prompt(query: str, candidates: list[dict], top_k: int) -> str:
    candidate_block = "\n\n".join(
        f"[{i}] {c['title']}\n{c['body'][:800]}" for i, c in enumerate(candidates)
    )
    return f"""Rank these candidate notes by relevance to the query below. The query is looking for cross-domain conceptual bridges — notes from different domains that share underlying structure.

Query: {query}

Candidates:
{candidate_block}

Respond with ONLY a JSON array of the top {top_k} candidate indices in order of relevance, like [3, 0, 7, 1, 12]. No prose, no code fence, just the JSON array."""


def rerank_via_cli(query: str, candidates: list[dict], top_k: int, cli: str, model: str | None) -> tuple[list[dict], dict]:
    prompt = build_rerank_prompt(query, candidates, top_k)
    resp = cli_invoke(cli, prompt, model=model, timeout=240)
    parsed = parse_json_response(resp.text)
    parse_error = None
    if isinstance(parsed, list) and all(isinstance(i, int) for i in parsed):
        try:
            reranked = [candidates[i] for i in parsed[:top_k]]
        except IndexError as e:
            parse_error = f"index out of range: {e}"
            reranked = candidates[:top_k]
    else:
        parse_error = f"failed to parse list of ints from {resp.text[:200]!r}"
        reranked = candidates[:top_k]
    trace = {
        "method": "cli",
        "cli": cli,
        "model": model,
        "prompt": prompt,
        "raw_response": resp.text,
        "parsed": parsed,
        "parse_error": parse_error,
        "wall_clock_seconds": resp.wall_clock_seconds,
    }
    return reranked, trace


def rerank_via_api(query: str, candidates: list[dict], top_k: int, model: str) -> tuple[list[dict], dict]:
    from anthropic import Anthropic
    aclient = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    prompt = build_rerank_prompt(query, candidates, top_k)
    msg = aclient.messages.create(
        model=model,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text.strip()
    parsed = parse_json_response(raw)
    parse_error = None
    if isinstance(parsed, list) and all(isinstance(i, int) for i in parsed):
        try:
            reranked = [candidates[i] for i in parsed[:top_k]]
        except IndexError as e:
            parse_error = f"index out of range: {e}"
            reranked = candidates[:top_k]
    else:
        parse_error = f"failed to parse list of ints from {raw[:200]!r}"
        reranked = candidates[:top_k]
    trace = {
        "method": "api",
        "api": "anthropic",
        "model": model,
        "prompt": prompt,
        "raw_response": raw,
        "parsed": parsed,
        "parse_error": parse_error,
    }
    return reranked, trace


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--llm", choices=["cli", "api"], default="cli",
                    help="Use subscription CLI (default) or paid API for rerank")
    ap.add_argument("--cli", choices=["claude", "codex", "gemini"], default="claude",
                    help="Which CLI to invoke when --llm=cli")
    ap.add_argument("--model", default=None,
                    help="Optional model override (depends on --llm choice)")
    args = ap.parse_args()

    notes = [json.loads(line) for line in args.corpus.read_text().splitlines() if line.strip()]
    queries = [json.loads(line) for line in args.queries.read_text().splitlines() if line.strip()]

    vclient = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    note_embeddings = embed_corpus(notes, vclient)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    traces_dir = args.out.parent / "traces" / args.out.stem
    traces_dir.mkdir(parents=True, exist_ok=True)

    api_model = args.model or "claude-sonnet-4-20250514"

    started = time.time()
    with args.out.open("w") as f:
        for q in queries:
            q_emb = np.array(vclient.embed([q["query_text"]], model=EMBED_MODEL, input_type="query").embeddings[0])
            scored = [(n, cosine(q_emb, note_embeddings[n["id"]])) for n in notes]
            candidates = sorted(scored, key=lambda x: x[1], reverse=True)
            candidate_notes = [n for n, _ in candidates]

            if args.llm == "cli":
                reranked, rerank_trace = rerank_via_cli(q["query_text"], candidate_notes, args.top_k, args.cli, args.model)
            else:
                reranked, rerank_trace = rerank_via_api(q["query_text"], candidate_notes, args.top_k, api_model)

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
                "cosine_top_5_pre_rerank": [{"id": n["id"], "score": s} for n, s in candidates[:5]],
                "rerank_trace": rerank_trace,
                "final_top_k": [{"id": n["id"], "title": n["title"]} for n in reranked],
            }, indent=2))
            print(f"  {q['query_id']}: top1={reranked[0]['id']}", flush=True)

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
