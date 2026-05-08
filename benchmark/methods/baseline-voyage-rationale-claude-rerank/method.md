# Baseline: Voyage-3 + rationale-augmented + Claude final rerank (Method 4b)

Splits the rationale-augmented baseline into a *cleaner* version of the kill-product test. Method 4a uses the *same Voyage embedding* to rank rationale-augmented candidates; Method 4b lets Claude read both query and augmented candidates and pick the top-K directly. Different signal channels, both worth scoring against geodesic retrieval.

## What this measures

Compared to Method 4a (`baseline-voyage-rationale-rerank`):
- 4a measures: does *appending an LLM-generated rationale to candidate text and re-embedding* surface better cross-domain bridges?
- 4b measures: does an LLM-as-judge, given query + rationale-augmented candidates, pick better cross-domain bridges than the same LLM working without the rationale step?

Either of these is a stronger baseline than plain Method 3. Method 6 (geodesic) needs to beat both.

## Inputs

- `../../corpus/notes.jsonl`
- `../../queries.public.jsonl`

## Outputs

- `../../results/voyage-rationale-claude-rerank-{date}.jsonl`
- `../../results/traces/voyage-rationale-claude-rerank-{date}/q<NN>.json`
- `../../results/traces/voyage-rationale-claude-rerank-{date}/manifest.json`

## Run

```bash
export VOYAGE_API_KEY=...
pip install -r requirements.txt
python run.py --corpus ../../corpus/notes.jsonl --queries ../../queries.public.jsonl \
              --out ../../results/voyage-rationale-claude-rerank-claude-$(date +%Y-%m-%d).jsonl
```

Same flags as Method 4a:
- `--llm cli|api` (default `cli`)
- `--cli claude|codex|gemini` (default `claude`)
- `--candidate-pool N` (default 0 = full corpus at v0.1)
- `--concurrency N` (default 5; rationale generation parallelism)

## Implementation

1. Embed corpus + query with `voyage-3` (cached).
2. Cosine-rank → take all 50 candidates per query (or `--candidate-pool` cap).
3. For each candidate, generate a one-paragraph bridge rationale (parallel via thread pool).
4. **Different from Method 4a**: send `query + (candidate_title + candidate_body + rationale)` for *all* candidates to the LLM in a single rerank prompt; LLM returns top-K indices.
5. Take top-5.

The candidate-pool cap matters more here than in 4a: 50 rationale-augmented blocks of 3-5 sentences each is ~10k tokens of input on the rerank step. Within Claude's context window but adds latency. With `--candidate-pool 15` it's ~3k tokens, faster per query.

## Why both 4a and 4b

The reviewer's point: the naming "Voyage-3 + Claude rerank with bridge-rationale generation" was ambiguous about what does the final rank. 4a does it with cosine; 4b does it with Claude. Splitting them lets the eventual results table attribute any lift to the right cause.
