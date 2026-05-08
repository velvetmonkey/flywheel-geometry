# geodesic-bench

Open benchmark for cross-domain semantic bridge retrieval in personal knowledge bases.

**Status:** v0.1 scaffold + 50-note synthetic corpus + 30 cross-domain queries + four baseline methods in place. v0.1 results land next; v0.2 scales corpus to 500 notes before any external claim.

## The question

When the *same idea* lives across multiple domains under different vocabulary, can activation-derived geometry surface those cross-domain bridges more reliably than a strong embedding pipeline that's been given the chance to *generate the bridge rationale itself*?

This is a narrower claim than "geodesic retrieval beats cosine for general search." Cosine wins on precision search. Geodesic mode is a candidate *additional* axis that answers "what's structurally adjacent across domains?" — and only earns its place if it beats the rationale-augmented embedding baseline by a meaningful margin.

## Setup

- **Corpus** (`corpus/notes.jsonl`): **50 fully synthetic notes** authored from a fictional persona (see [`corpus/PERSONA.md`](./corpus/PERSONA.md)). Spans 10 domains (AI research, equine training, quant finance, vehicle/equipment, philosophy, travel, career, software architecture, home admin, daily journal). Designed so the same underlying concepts (feedback loops, regime change, calibration, tail dependence, exploration vs exploitation, attention texture) recur across surface-different domains. v0.1 size; **v0.2 scales to 500 notes** for the public claim.
- **Queries** (`queries.public.jsonl`): 30 hand-written queries — `query_id`, `query_text`, `is_control` only. 27 are cross-domain bridge queries (the headline benchmark). 3 are intra-domain control queries (sanity floor; reported separately, never mixed into the headline metric).
- **Gold targets** (`gold/targets.jsonl`): hidden target note IDs + target domains + rationale per query. **Method runners must not read this file.** Only the evaluator reads `gold/`. See [`gold/README.md`](./gold/README.md) for the rationale.
- **Evaluation**: blind precision@5 and nDCG@5 against hidden gold targets; head-to-head human + LLM-as-judge preference under hidden method labels.

## Methods

The benchmark separates four signal channels deliberately — **embedding quality**, **reranker quality**, **LLM-explanation effect**, and **activation geometry** — so we can attribute any observed lift to the right cause.

| # | Method | Tests | Status | Path |
|---|---|---|---|---|
| 1 | BM25 | lexical sanity floor | scaffolded | [`methods/baseline-bm25/`](./methods/baseline-bm25/) |
| 2 | Voyage-3 + **Voyage native reranker** | embedding + reranker quality, no LLM | scaffolded | [`methods/baseline-voyage-native-rerank/`](./methods/baseline-voyage-native-rerank/) |
| 3 | Voyage-3 + Claude rerank | + LLM reranker effect | scaffolded | [`methods/baseline-voyage-rerank/`](./methods/baseline-voyage-rerank/) |
| 4a | Voyage-3 + **rationale-augmented embedding rerank** *(kill-product test)* | + LLM explanation effect, embedding-only re-rank | scaffolded | [`methods/baseline-voyage-rationale-rerank/`](./methods/baseline-voyage-rationale-rerank/) |
| 4b | Voyage-3 + rationale-augmented + **Claude final rerank** | + LLM explanation effect, LLM as final judge | scaffolded | [`methods/baseline-voyage-rationale-claude-rerank/`](./methods/baseline-voyage-rationale-claude-rerank/) |
| 5 | @slashreboot's introspective coordinate probe | learned-discourse priors | v0.2 | — |
| 6 | **Activation extraction via TransformerLens + kNN** *(the real hypothesis)* | activation-space geometry | v0.2 | [`methods/method-geodesic/`](./methods/method-geodesic/) (spec only) |

A frontier-commercial baseline using `voyage-4-large + rerank-2.5` lives at [`methods/baseline-voyage-frontier/`](./methods/baseline-voyage-frontier/). Treat it as a moving target — Voyage's API rolls forward — rather than a stable v0.1 comparison point. The stable comparison anchors are Methods 1–4b, which all pin `voyage-3` and `rerank-2`.

Methods 4a + 4b are the bar Method 6 must clear. Comparison logic:
- Method 2 vs Method 3 → isolates "Claude as reranker" from "Voyage native reranker" (no rationale in either).
- Method 3 vs Method 4a → isolates the *rationale-generation* effect, with the *same* Voyage embedding step doing the final rank.
- Method 4a vs Method 4b → isolates the effect of the *final reranker* (Voyage-cosine over augmented text, vs Claude reading both query and augmented text and picking).
- Method 6 vs Method 4b → activation geometry vs the strongest LLM-augmented embedding baseline.

### LLM dispatch: subscription CLI by default

Methods 3 and 4 invoke an LLM many times. By default each runner shells out to a local subscription CLI (`claude`, `codex`, or `gemini`) — same pattern as the [Roundtable MCP server](https://github.com/velvetmonkey/roundtable). This piggybacks on the user's AI subscriptions instead of burning per-token API credits.

- **`--llm cli` (default)** — `--cli claude` / `--cli codex` / `--cli gemini`. CLI startup overhead is non-trivial (~10–25s/call); `--concurrency N` runs N parallel calls. Subscription tier dictates how high `N` can go before rate-limiting.
- **`--llm api`** — falls back to the Anthropic SDK with `ANTHROPIC_API_KEY`. No rate-limit ceiling but billed per-token. Useful for high-volume runs or reproducibility against published API model digests.

Method 4 (rationale generation) is the dominant cost. At v0.1 scale that's 1,500 LLM calls per run; with `--candidate-pool 15` it drops to 450. Configure both flags per run; both are logged in the manifest for reproducibility.

### Candidate pool sizing (v0.1)

For the v0.1 50-note corpus, methods 3 and 4 rerank **the full corpus** (k=50) per query rather than a top-K cosine slice. Reason: at this scale, top-K cosine slicing risks excluding genuinely structurally-adjacent candidates that cosine ranks low — exactly the case Method 4 is supposed to test. At v0.2 scale (500 notes), candidate pools widen to top-100 or top-200 to balance compute vs coverage; the exact pool size will be pre-registered before v0.2 results run.

## Pre-registered geodesic method (Method 6, v0.2)

To ship a credible activation-geometry comparison, the geodesic method's spec needs to be locked **before** results are scored. v0.2 will commit the following before running:

- **Model**: one open-source language model (Llama 3 70B *or* Gemma 3 27B; pinned by digest, single layer chosen).
- **Layer**: a single residual-stream layer chosen via prior held-out probe-quality work (not chosen on test corpus).
- **Pooling**: last-token pooling (default) or mean-pooling (ablation); both reported.
- **Distance**: Euclidean in activation space, then graph distance via shortest-path on a kNN graph (k=10 by default; k swept as ablation).
- **Optional alternatives**: diffusion distance on the kNN graph; geodesic distance via principal-curve fitting where the manifold structure is known. Ablations only — mainline is shortest-path.
- **No magic 3D projection**: UMAP appears only in visualisations, never in the retrieval pipeline.

The pre-registered spec lives at `methods/method-geodesic/method.md` (v0.2). Any deviation between pre-registered spec and run gets logged as a protocol violation.

## Reproducibility

Every method submission must save:

- Top-5 ranked results per query (the scoring artifact).
- **Full request/response traces** per query in `results/traces/<method>-<date>/q<NN>.json` — including: model IDs (with pinned digest where applicable), prompts verbatim, temperature, candidate pool size, raw LLM responses, parse errors, score traces.
- A `manifest.json` per run capturing: corpus checksum, queries checksum, env summary (Python version, key dependency versions), random seeds, wall-clock runtime.

Pinned model IDs are not enough — serving infrastructure can change underneath them. The full trace is the unit of reproducibility.

## Decision criterion

**Primary metric**: precision@5 on the **27 cross-domain bridge queries**.
**Secondary metric**: precision@5 on the **3 intra-domain control queries**, reported separately.

Method 6 (activation-derived geodesic retrieval) ships only if it shows ≥20% precision@5 lift over **all three of Method 3, Method 4a, and Method 4b** on the **primary** metric (27 cross-domain bridge queries). Control queries are reported alongside but never mixed into the headline.

Equal or worse vs Method 4a *or* Method 4b on the primary metric means the bridge "effect" is the LLM explaining adjacency, not the geometry surfacing it. Project pivots — to bridge-tension via relational structure, to direct activation extraction without the introspective wrapper, or away from the manifold thesis entirely. The pivot post is the launch.

## Scoring runs

After a method writes `results/<method>-<date>.jsonl` + traces, score it with:

```bash
python eval.py --gold gold/targets.jsonl --queries queries.public.jsonl \
  results/<method>-<date>.jsonl [results/<method2>-<date>.jsonl ...]
```

The evaluator never reads anything under `methods/` — only `gold/`, `queries.public.jsonl`, `corpus/notes.jsonl`, and the named results files. **Method runners never read `gold/`** (enforced by the gold-leak tripwire below).

### Pre-scoring guards

Every results file is validated before metrics are computed. Any failure exits non-zero with no row appended to `RESULTS.md`. Guards:

1. Each results file has exactly 30 rows.
2. No missing or duplicated `query_id` within a file.
3. No duplicate note IDs within any `top_k` array.
4. Every retrieved ID exists in `corpus/notes.jsonl`.
5. **Gold-leak tripwire**: no method output contains `target_note_ids`, `target_domains`, or `rationale` fields.

### v0.1 tie threshold (debugging aid only)

For v0.1 harness validation, methods within **0.05 absolute** primary p@5 of the best are flagged `tied_with_best: yes` in the eval table. **This is a debugging aid, not the v0.2 ship criterion.** The real ship criterion remains the ≥20% lift stated in the **Decision criterion** above. Use the v0.1 tie marker to surface methods that are statistically indistinguishable at small N (27 primary queries); use the ≥20% bar to make ship/pivot decisions.

Per-query traces with `hits_at_5` boolean arrays land at `results/<method>-<date>-eval.json`.

## Submitting a method

PRs welcome. A method submission is a directory under `methods/` containing:

- `method.md` — short description + pre-registered theoretical justification
- `run.py` — deterministic script that reads `corpus/notes.jsonl` + `queries.public.jsonl` and writes both `results/<method>-<date>.jsonl` and `results/traces/<method>-<date>/`
- `requirements.txt` — Python deps (or equivalent for other languages)

Open an issue first if you want to discuss corpus, query set, or method-comparison design — the goal is a stable comparison point, not a moving target.

## Why this exists

Lubana et al. ([Goodfire AI, 2025–2026](https://www.youtube.com/watch?v=F9eEYWX64ZA)) named the tool gap: *"something like a SAE but which respects nonlinear geometry."* The Goodfire material proves activation-space manifolds align with behavior-space manifolds — that geometry-respecting interventions outperform linear ones for *control*. It does not yet prove that geodesic distance helps *retrieval*. That translation is what this benchmark tests.

External technical review (2026-05-08) raised the kill-product baseline (Method 4) as the test that distinguishes *retrieval* from *presentation*. If rationale-augmented embeddings tie or beat the geodesic method on cross-domain bridges, the bridges were always discoverable by strong embedding + LLM explanation, and the manifold framing was carrying weight it didn't earn.
