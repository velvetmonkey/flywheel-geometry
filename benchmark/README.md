# geodesic-bench

Open benchmark for cross-domain semantic bridge retrieval in personal knowledge bases.

**Status:** v0.1 scaffold. Three baseline methods are in place; corpus + queries land next; first results land with v0.2.

## The question

Does retrieval over manifold-derived coordinates surface useful cross-domain bridges that strong embedding retrieval misses — or is it cosine similarity in a new costume?

## Setup

- **Corpus** (`corpus/notes.jsonl`): 50 anonymised personal-vault notes spanning ≥5 domains (technical projects, personal/operational, philosophical, AI research, daily/temporal).
- **Queries** (`queries.jsonl`): 30 cross-domain bridge-finding queries with hidden gold-standard target notes.
- **Evaluation**: blind precision@5 and nDCG@5; head-to-head human + LLM-as-judge preference under hidden method labels.

## Methods (three baselines in place; geodesic method is the v0.2 hypothesis)

| Method | Status | Path |
|---|---|---|
| BM25 (lexical sanity floor) | scaffolded | [`methods/baseline-bm25/`](./methods/baseline-bm25/) |
| Voyage-3 + Claude rerank (strong embedding baseline) | scaffolded | [`methods/baseline-voyage-rerank/`](./methods/baseline-voyage-rerank/) |
| Voyage-3 + Claude rerank **with bridge-rationale generation** *(kill-product test)* | scaffolded | [`methods/baseline-voyage-rationale-rerank/`](./methods/baseline-voyage-rationale-rerank/) |
| Activation extraction via TransformerLens (the actual hypothesis) | v0.2 | — |
| @slashreboot's introspective coordinate probe (cheap baseline) | v0.2 | — |

## Decision criterion

Geodesic (activation-derived) retrieval continues only if it shows ≥20% precision@5 lift over **both** baseline 2 (Voyage + rerank) and baseline 3 (Voyage + rationale rerank) on cross-domain bridge queries.

Equal or worse vs baseline 3 means the manifold "effect" is the LLM explaining adjacency, not the geometry surfacing it. Project pivots accordingly — to bridge-tension via relational structure, or to direct activation extraction without the introspective probe wrapper, or away from the manifold thesis entirely.

## Submitting a method

PRs welcome. A method submission is a directory under `methods/` containing:

- `method.md` — short description + theoretical justification
- `run.py` — deterministic script that takes `corpus/notes.jsonl` + `queries.jsonl` and emits ranked top-5 per query into `results/<method>-<date>.jsonl`
- `requirements.txt` — Python deps (or equivalent for other languages)

Open an issue first if you want to discuss the corpus or query set — the goal is a stable comparison point, not a moving target.

## Why this exists

Lubana et al. ([Goodfire AI, 2025–2026](https://www.youtube.com/watch?v=F9eEYWX64ZA)) named the tool gap: *"something like a SAE but which respects nonlinear geometry."* This is one falsifiable test of whether a geodesic-respecting retriever clears the bar that strong embeddings already set on personal-knowledge-base retrieval.

Codex's round-2 critique (logged in [`projects/flywheel-geometry/flywheel-geometry-council-2026-05-07.md`](https://github.com/velvetmonkey) in the upstream vault) elevated to baseline #3: if rationale-augmented embeddings tie or beat the manifold method, the "bridge-finding" effect is presentation, not retrieval.
