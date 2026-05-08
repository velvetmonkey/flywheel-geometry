# CLI-direct rerank — pure-LLM baseline, no embedding

Sends `query + all 50 corpus notes` to a single subscription-CLI call (`claude` / `codex` / `gemini`), asks for the top-5 most relevant notes. **No embedding step**, no API budget — pure subscription dispatch.

## What this measures

How well does a frontier LLM, given the **entire corpus** in context, identify cross-domain conceptual bridges? At v0.1 scale (50 notes, ~25K tokens of corpus) the whole vault fits in a single prompt; no pre-filter is needed.

This is the simplest possible LLM-only baseline. If the geodesic method (Method 6, v0.2) doesn't beat this, the geometry framing isn't earning its place.

## Why this exists alongside the Voyage methods

The Voyage-track methods (2, 3, 4a, 4b) require `VOYAGE_API_KEY` — paid API budget for embeddings + native reranking. The CLI-only methods are subscription-only: zero per-call cost, just CLI dispatch. Functionally, at v0.1 scale (50 notes, full-corpus reranking) the Voyage cosine pre-filter does no real work — the LLM rerank step sees the full corpus anyway. Cosine pre-filtering only matters at v0.2 scale (500+ notes) where context budget forces a top-K candidate pool.

The Voyage-track methods stay in the repo as the pre-registered v0.2 anchors. The CLI-only methods are how we run v0.1 cheaply on subscriptions.

| Voyage equivalent | This method |
|---|---|
| 3: voyage-rerank | cli-direct-rerank (this) |
| (no equivalent) | cli-rationale-rerank |

## Inputs

- `../../corpus/notes.jsonl` — all 50 notes
- `../../queries.public.jsonl` — 30 queries

## Outputs

- `../../results/cli-direct-rerank-{cli}-{date}.jsonl`
- `../../results/traces/cli-direct-rerank-{cli}-{date}/q<NN>.json`
- `../../results/traces/cli-direct-rerank-{cli}-{date}/manifest.json`

## Run

```bash
pip install -r requirements.txt
python run.py --corpus ../../corpus/notes.jsonl --queries ../../queries.public.jsonl \
              --out ../../results/cli-direct-rerank-claude-$(date +%Y-%m-%d).jsonl \
              --cli claude

# Faster CLI (~3× faster on smoke tests):
python run.py ... --cli codex
python run.py ... --cli gemini
```

## Implementation

For each of the 30 queries:
1. Build a prompt: query + all 50 corpus notes (truncated bodies to fit context).
2. Single CLI call: ask the model to return a JSON array of the top-5 note IDs.
3. Parse the response, write the top-5 to JSONL + full trace to traces/.

Prompt size: ~30K tokens for the corpus block (50 notes × ~600 chars/note). All three CLIs handle this comfortably (Claude 200K, Codex 200K, Gemini 1M).

Wall-clock estimate: 30 × ~25s (claude) or ~10s (codex) per call = 5-15 min total.
