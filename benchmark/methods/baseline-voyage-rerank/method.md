# Baseline: Voyage-3 + Claude rerank

LLM-as-reranker baseline. Isolates the *LLM-rerank effect* on top of strong embedding retrieval — without prompting the LLM to generate any rationale or explanation.

## What this measures

How far retrieval quality improves when a frontier LLM (Claude Sonnet 4.6) is allowed to rerank embedding-retrieved candidates **without** generating bridge rationales. Compared against:
- Method 2 (Voyage native rerank, no LLM) → isolates "Claude as reranker" vs "Voyage's native reranker."
- Method 4 (rerank with rationale generation) → isolates the *explanation-generation* effect from raw rerank.

## Inputs

- `../../corpus/notes.jsonl`
- `../../queries.public.jsonl`

## Outputs

- `../../results/voyage-rerank-{date}.jsonl` — top-5 reranked note IDs per query
- `../../results/traces/voyage-rerank-{date}/q<NN>.json` — full request/response trace per query (prompt, raw response, parse path/errors)
- `../../results/traces/voyage-rerank-{date}/manifest.json` — run metadata (model IDs, corpus/query checksums, wall-clock)

## Run

Default uses the **subscription CLI** (`claude` / `codex` / `gemini`) — piggy-backs on
your AI subscription rather than burning per-call API credits. Falls back to the
Anthropic API when `--llm api` is passed.

```bash
export VOYAGE_API_KEY=...                # always needed for embeddings
# only needed for --llm api:
# export ANTHROPIC_API_KEY=...
pip install -r requirements.txt

# Subscription CLI (default — claude); 30 calls × ~20s ≈ 10 min:
python run.py --corpus ../../corpus/notes.jsonl --queries ../../queries.public.jsonl \
              --out ../../results/voyage-rerank-claude-$(date +%Y-%m-%d).jsonl

# Try a different CLI:
python run.py ... --cli codex     # ~3× faster than claude in our smoke tests
python run.py ... --cli gemini

# Fall back to API:
python run.py ... --llm api --model claude-sonnet-4-20250514
```

## Implementation

1. Embed each note with `voyage-3` (input_type=document).
2. Embed each query with `voyage-3` (input_type=query).
3. Cosine-rank all 50 candidates per query (full corpus at v0.1; top-100 at v0.2 scale).
4. LLM rerank: send query + ALL candidate titles+bodies to the chosen model with a "rank these by relevance" prompt. The prompt explicitly mentions cross-domain bridges to mirror the evaluation criterion. Parse the JSON array of top-K indices.
5. Write top-5 to JSONL; write full trace (prompt + raw response + parse path) to traces/.

Caching: Voyage embeddings cached by note ID + model version under `.cache/`.

## What this baseline doesn't test

- Whether prompted rationale-generation helps. That's Method 4 (the kill-product test).
- Whether activation-space geometry surfaces things no embedding sees. That's Method 6 (v0.2).
