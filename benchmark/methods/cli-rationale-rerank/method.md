# CLI rationale-rerank — kill-product test, no embedding

For each query, generate a one-paragraph bridge rationale per candidate via subscription CLI (in parallel), then send query + all rationale-augmented candidates to a single CLI rerank call. **No embedding step**, no API budget — pure subscription.

This is the no-Voyage equivalent of [`baseline-voyage-rationale-claude-rerank/`](../baseline-voyage-rationale-claude-rerank/) (Method 4b). Same kill-product signal channel: gives a frontier LLM full opportunity to *manufacture* a relevance rationale for each candidate, then judge top-K. If Method 6 (geodesic, v0.2) doesn't beat this, the bridge effect is presentation, not retrieval.

## What this measures

Per query, can the LLM:
1. Generate a plausible bridge-rationale for each of 50 corpus notes (50 calls, parallel).
2. Then rank the rationale-augmented candidates by relevance to the query (1 call).

If the answer is "yes — the rationale-augmented top-5 matches gold targets at high precision," then the bridge effect is in the *explanation step*, not in any underlying geometry. Geodesic retrieval has to clear this bar to earn its place.

## Cost knobs

Default `--candidate-pool 0` runs the rationale step over the **full corpus** (50 candidates × 30 queries = 1,500 LLM calls + 30 rerank calls = 1,530 total).

For smoke / debug runs use `--candidate-pool 5` to only generate rationales for the top-5 by lexical overlap (uses BM25-style word-overlap pre-filter — no embedding API needed). That's 30 × 5 = 150 rationale calls + 30 rerank = 180 total.

| Mode | Calls | Wall-clock @claude (--concurrency 5) | Wall-clock @codex |
|---|---|---|---|
| Smoke (--candidate-pool 5) | 180 | ~12 min | ~5 min |
| Full (--candidate-pool 0) | 1,530 | ~1.7 h | ~40 min |

## Run

```bash
pip install -r requirements.txt

# Smoke run first (recommended):
python run.py --corpus ../../corpus/notes.jsonl --queries ../../queries.public.jsonl \
              --out ../../results/cli-rationale-rerank-claude-smoke-$(date +%Y-%m-%d).jsonl \
              --cli claude --candidate-pool 5

# Full run after smoke validates:
python run.py ... --cli claude --candidate-pool 0 --concurrency 8
```

Flags:
- `--cli claude|codex|gemini` (default claude)
- `--candidate-pool N` (default 0 = full corpus)
- `--concurrency N` (default 5; tune up to ~10 before subscription rate limits)
- `--model MODEL` (optional CLI model override)

## Implementation

For each query:

1. **Candidate selection**: if `--candidate-pool 0`, all 50 notes; otherwise top-N by simple BM25-style word overlap with the query. (Lightweight Python tokenization, no external API.)
2. **Rationale generation**: thread-pool of `--concurrency` parallel CLI calls — each generates a 3-5 sentence bridge rationale linking one candidate to the query.
3. **Final rerank**: single CLI call sees query + (title + body + rationale) for each candidate, returns top-K JSON array of note IDs.
4. Write top-5 to JSONL + full trace per query (every rationale prompt + response, the rerank prompt + response).

Pre-filter by word overlap is intentionally crude: it's how `--candidate-pool` saves money when a Voyage cosine pre-filter isn't available. At v0.2 scale, the embedding pre-filter goes back in.
