# geodesic-bench

Open benchmark for cross-domain semantic bridge retrieval in personal knowledge bases.

**Status:** v0.1 scaffold + 50-note synthetic corpus + 30 cross-domain queries + four baseline methods in place. v0.1 results land next; v0.2 scales corpus to 500 notes before any external claim.

## The question

When the *same idea* lives across multiple domains under different vocabulary, can activation-derived geometry surface those cross-domain bridges more reliably than a strong embedding pipeline that's been given the chance to *generate the bridge rationale itself*?

This is a narrower claim than "geodesic retrieval beats cosine for general search." Cosine wins on precision search. Geodesic mode is a candidate *additional* axis that answers "what's structurally adjacent across domains?" — and only earns its place if it beats the rationale-augmented embedding baseline by a meaningful margin.

## Setup

- **Corpus** (`corpus/notes.jsonl`): **50 fully synthetic notes** authored from a fictional persona (see [`corpus/PERSONA.md`](./corpus/PERSONA.md)). Spans 10 domains (AI research, equine training, quant finance, vehicle/equipment, philosophy, travel, career, software architecture, home admin, daily journal). Designed so the same underlying concepts (feedback loops, regime change, calibration, tail dependence, exploration vs exploitation, attention texture) recur across surface-different domains. v0.1 size; **v0.2 scales to 500 notes** for the public claim.
- **Queries** (`queries.jsonl`): 30 hand-written cross-domain bridge queries with hidden gold target note IDs. 27 cross-domain, 3 intra-domain control queries.
- **Evaluation**: blind precision@5 and nDCG@5 against hidden gold targets; head-to-head human + LLM-as-judge preference under hidden method labels.

## Methods

The benchmark separates four signal channels deliberately — **embedding quality**, **reranker quality**, **LLM-explanation effect**, and **activation geometry** — so we can attribute any observed lift to the right cause.

| # | Method | Tests | Status | Path |
|---|---|---|---|---|
| 1 | BM25 | lexical sanity floor | scaffolded | [`methods/baseline-bm25/`](./methods/baseline-bm25/) |
| 2 | Voyage-3 + **Voyage native reranker** | embedding + reranker quality, no LLM | scaffolded | [`methods/baseline-voyage-native-rerank/`](./methods/baseline-voyage-native-rerank/) |
| 3 | Voyage-3 + Claude rerank | + LLM reranker effect | scaffolded | [`methods/baseline-voyage-rerank/`](./methods/baseline-voyage-rerank/) |
| 4 | Voyage-3 + Claude rerank **with bridge-rationale generation** *(kill-product test)* | + LLM explanation effect | scaffolded | [`methods/baseline-voyage-rationale-rerank/`](./methods/baseline-voyage-rationale-rerank/) |
| 5 | @slashreboot's introspective coordinate probe | learned-discourse priors | v0.2 | — |
| 6 | **Activation extraction via TransformerLens + kNN** *(the real hypothesis)* | activation-space geometry | v0.2 | — |

Method 4 is the bar Method 6 must clear. Method 2 vs Method 3 isolates "LLM-as-reranker" effects from "embedding + reranker" baseline. Method 3 vs Method 4 isolates the *explanation-generation* effect from the rerank.

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

Method 6 (activation-derived geodesic retrieval) ships only if it shows ≥20% precision@5 lift over **both** Method 3 (Voyage + Claude rerank) and Method 4 (Voyage + Claude rerank with rationale generation) on the 30 cross-domain bridge queries.

Equal or worse vs Method 4 means the bridge "effect" is the LLM explaining adjacency, not the geometry surfacing it. Project pivots — to bridge-tension via relational structure, to direct activation extraction without the introspective wrapper, or away from the manifold thesis entirely. The pivot post is the launch.

## Submitting a method

PRs welcome. A method submission is a directory under `methods/` containing:

- `method.md` — short description + pre-registered theoretical justification
- `run.py` — deterministic script that reads `corpus/notes.jsonl` + `queries.jsonl` and writes both `results/<method>-<date>.jsonl` and `results/traces/<method>-<date>/`
- `requirements.txt` — Python deps (or equivalent for other languages)

Open an issue first if you want to discuss corpus, query set, or method-comparison design — the goal is a stable comparison point, not a moving target.

## Why this exists

Lubana et al. ([Goodfire AI, 2025–2026](https://www.youtube.com/watch?v=F9eEYWX64ZA)) named the tool gap: *"something like a SAE but which respects nonlinear geometry."* The Goodfire material proves activation-space manifolds align with behavior-space manifolds — that geometry-respecting interventions outperform linear ones for *control*. It does not yet prove that geodesic distance helps *retrieval*. That translation is what this benchmark tests.

External technical review (2026-05-08) raised the kill-product baseline (Method 4) as the test that distinguishes *retrieval* from *presentation*. If rationale-augmented embeddings tie or beat the geodesic method on cross-domain bridges, the bridges were always discoverable by strong embedding + LLM explanation, and the manifold framing was carrying weight it didn't earn.
