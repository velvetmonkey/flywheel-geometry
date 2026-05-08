# Baseline: Voyage-3 + Claude rerank with bridge-rationale generation

**The kill-product test.** Codex's round-2 #6 critique elevated to a tracked baseline — and per external review, candidate pool widened to the full corpus at v0.1 scale to give the rationale-augmented method maximum chance to rescue distant bridge candidates.

## What this measures

Same retrieval pipeline as `baseline-voyage-rerank`, but at rerank time the LLM **also** generates a bridge rationale — a short paragraph explaining how each candidate connects to the query — *before* the candidate's text gets re-embedded and re-ranked.

If raters prefer this baseline's results at equal or higher rates than plain rerank, the bridge "effect" the geodesic method claims to surface is **the LLM explaining adjacency, not the geometry surfacing it**. The product premise dies.

If geodesic retrieval (Method 6, v0.2) can't beat *both* plain rerank AND rationale-augmented rerank by ≥20% precision@5, the project pivots — to bridge-tension via relational structure (Codex's reframe), to direct activation extraction without the introspective wrapper, or away from the manifold thesis entirely.

## Inputs

- `../../corpus/notes.jsonl`
- `../../queries.jsonl`

## Outputs

- `../../results/voyage-rationale-rerank-{date}.jsonl` — top-5 reranked + per-candidate rationales + rescore values
- `../../results/traces/voyage-rationale-rerank-{date}/q<NN>.json` — full trace per query: every rationale prompt + response, cosine before and after, rescore deltas
- `../../results/traces/voyage-rationale-rerank-{date}/manifest.json` — run metadata

## Run

Default uses the **subscription CLI** (`claude` / `codex` / `gemini`). Rationale
generation is the most expensive step in the benchmark — full-corpus reranking
at v0.1 scale = 50 candidates × 30 queries = **1,500 LLM calls per run**.
At ~20s/call sequential against `claude`, that's ~8h. Concurrency knocks it down:

| --concurrency | Wall-clock |
|---|---|
| 1 (sequential) | ~8h |
| 5 (default) | ~1.7h |
| 10 | ~50m |
| 20 | ~25m, but you'll hit subscription rate limits |

```bash
export VOYAGE_API_KEY=...
# only needed for --llm api:
# export ANTHROPIC_API_KEY=...
pip install -r requirements.txt

# Subscription CLI (default), 5 parallel:
python run.py --corpus ../../corpus/notes.jsonl --queries ../../queries.jsonl \
              --out ../../results/voyage-rationale-rerank-claude-$(date +%Y-%m-%d).jsonl

# Faster CLI (codex was ~3× faster than claude in smoke tests):
python run.py ... --cli codex --concurrency 8

# Reduce candidate pool if rate-limited or impatient:
python run.py ... --candidate-pool 15   # 30q × 15 = 450 calls instead of 1500

# Fall back to paid API (no rate limits to speak of, billed per-token):
python run.py ... --llm api --model claude-sonnet-4-20250514
```

## Implementation

1. Embed corpus + queries with `voyage-3`.
2. Cosine-rank → take **all 50 candidates per query** at v0.1 scale (full corpus by default; cap with `--candidate-pool`). The wider-than-strictly-needed candidate pool is deliberate: the rationale-augmented method has to be given the same chance of rescuing genuinely distant bridge candidates that the geodesic method (Method 6) will get. Top-K cosine slicing would unfairly exclude the candidates we most need to see whether rationale-generation can pull up.
3. For each candidate, ask the chosen LLM to generate a one-paragraph bridge rationale: "How does this note connect to the query? What's the cross-domain insight?" Calls run in a thread pool at `--concurrency` parallelism.
4. Append rationale to candidate text, re-embed candidate-with-rationale via `voyage-3`.
5. Re-rank by cosine similarity to the query embedding.
6. Take top-5.

This is a deliberately strong baseline — the LLM gets full opportunity to manufacture a rationale that *makes* the candidate look relevant. If geodesic geometry's "find structurally-adjacent things you couldn't have found via keywords" claim is real, it should beat this. If not, the LLM was always doing the work.

## v0.2 candidate pool

At v0.2 scale (500 notes), full-corpus rationale generation is 10× the rationale calls per query — too expensive to be the default. Candidate pool will widen to top-100 from cosine, pre-registered before any results run. The 100-vs-50 question itself is an ablation worth running once: does the rationale-augmented baseline lose meaningful rescue capacity when the pool drops from 100 to 50?
