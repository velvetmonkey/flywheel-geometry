"""Voyage-3 + LLM rerank WITH bridge-rationale generation (Codex #6 / kill-product test).

Default mode: --llm cli (uses subscription tooling — claude / codex / gemini).
Optional: --llm api (Anthropic SDK + ANTHROPIC_API_KEY).

For each candidate, generate a bridge rationale via the chosen LLM; append rationale
to candidate text; re-embed augmented text; re-rank by cosine to query.

Concurrency: rationale generation runs in parallel via thread pool. Default 5
concurrent calls; tune via --concurrency. CLI startup overhead per call is ~10-25s,
so concurrency materially helps wall-clock on large candidate pools — but also
tests subscription rate limits faster.
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
from _lib.cli_invoker import cli_invoke, cli_invoke_many  # noqa: E402

CACHE_DIR = Path(__file__).parent / ".cache"
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


def build_rationale_prompt(query: str, note: dict) -> str:
    return f"""How does this note connect to the query? What's the cross-domain insight or conceptual bridge?

Query: {query}

Note title: {note['title']}
Note body: {note['body'][:1500]}

Write a one-paragraph rationale (3-5 sentences) explaining the connection. Be concrete about the bridge — name the underlying structure both share."""


def generate_rationales_via_cli(query: str, candidates: list[dict], cli: str, model: str | None, concurrency: int) -> list[tuple[str, dict]]:
    items = [(query, n) for n in candidates]

    def invoker(item: tuple[str, dict]):
        q, n = item
        prompt = build_rationale_prompt(q, n)
        return cli_invoke(cli, prompt, model=model, timeout=240)

    responses = cli_invoke_many(items, invoker, max_concurrency=concurrency)
    return [(r.text, {
        "method": "cli",
        "cli": r.cli,
        "model": r.model,
        "prompt": build_rationale_prompt(query, n),
        "rationale": r.text,
        "wall_clock_seconds": r.wall_clock_seconds,
    }) for r, n in zip(responses, candidates)]


def generate_rationales_via_api(query: str, candidates: list[dict], model: str) -> list[tuple[str, dict]]:
    from anthropic import Anthropic
    aclient = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    out = []
    for n in candidates:
        prompt = build_rationale_prompt(query, n)
        msg = aclient.messages.create(
            model=model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        rationale = msg.content[0].text.strip()
        out.append((rationale, {
            "method": "api",
            "api": "anthropic",
            "model": model,
            "prompt": prompt,
            "rationale": rationale,
        }))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--candidate-pool", type=int, default=0,
                    help="Cap rationale candidates to top-N cosine. 0 = full corpus (v0.1 default).")
    ap.add_argument("--llm", choices=["cli", "api"], default="cli",
                    help="Use subscription CLI (default) or paid API")
    ap.add_argument("--cli", choices=["claude", "codex", "gemini"], default="claude",
                    help="Which CLI to invoke when --llm=cli")
    ap.add_argument("--model", default=None, help="Optional model override")
    ap.add_argument("--concurrency", type=int, default=5,
                    help="Parallel CLI calls per query for rationale generation")
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
            q_started = time.time()
            q_emb = np.array(vclient.embed([q["query_text"]], model=EMBED_MODEL, input_type="query").embeddings[0])
            scored = [(n, cosine(q_emb, note_embeddings[n["id"]])) for n in notes]
            candidates_full = sorted(scored, key=lambda x: x[1], reverse=True)
            cap = args.candidate_pool if args.candidate_pool > 0 else len(notes)
            candidate_notes = [n for n, _ in candidates_full[:cap]]

            if args.llm == "cli":
                pairs = generate_rationales_via_cli(q["query_text"], candidate_notes, args.cli, args.model, args.concurrency)
            else:
                pairs = generate_rationales_via_api(q["query_text"], candidate_notes, api_model)
            rationales = [p[0] for p in pairs]
            rationale_traces = [{"note_id": n["id"], **t} for n, (_, t) in zip(candidate_notes, pairs)]

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
                "candidate_pool_size": len(candidate_notes),
                "cosine_top_5_pre_rationale": [{"id": n["id"], "score": s} for n, s in candidates_full[:5]],
                "rationales": rationale_traces,
                "rescored_top_k": [{"id": n["id"], "rationale": r, "rescore": s} for n, r, s in reranked],
                "query_wall_clock_seconds": round(time.time() - q_started, 2),
            }, indent=2))
            print(f"  {q['query_id']}: {len(candidate_notes)} rationales, {round(time.time() - q_started, 1)}s, top1={reranked[0][0]['id']}", flush=True)

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
        "corpus_checksum": file_checksum(args.corpus),
        "corpus_count": len(notes),
        "queries_checksum": file_checksum(args.queries),
        "queries_count": len(queries),
        "candidate_pool_cap": args.candidate_pool,
        "top_k": args.top_k,
    }
    (traces_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(queries)} results to {args.out}; traces in {traces_dir}; total {manifest['wall_clock_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
