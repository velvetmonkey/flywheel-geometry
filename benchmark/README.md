# geodesic-bench

Open benchmark for cross-domain semantic bridge retrieval in personal knowledge bases.

**Status:** stub. First run shipping week of 2026-05-19 alongside `flywheel-geometry` v0.2.

## The question

Does retrieval over manifold-derived coordinates surface useful cross-domain bridges that strong embedding retrieval misses — or is it cosine similarity in a new costume?

## Setup

- **Corpus**: 500 anonymised personal-vault notes, mixed-domain (technical, personal, philosophical, operational).
- **Queries**: 30 cross-domain bridge-finding queries with human-rated relevance labels.
- **Evaluation**: blind precision@5, blind nDCG@5, head-to-head human preference under hidden method labels.

## Baselines (initial)

- BM25
- Voyage-3 + LLM rerank
- Voyage-3 + LLM rerank with bridge-rationale generation *(distinguishes retrieval from presentation)*
- Cohere-embed-v4 + rerank
- OpenAI text-embedding-3 + rerank
- [@slashreboot](https://x.com/slashreboot)'s introspective coordinate probe (frontier model self-report)
- Layer-wise activation extraction (TransformerLens, open-source model)

## Decision criterion

Manifold-aware retrieval continues only if it shows ≥20% precision@5 lift over **both** baseline (2) and baseline (3) on cross-domain bridge queries. Equal or worse against rationale-augmented embeddings means the manifold "effect" is the LLM explaining adjacency, not the geometry surfacing it. Project pivots accordingly.

## Submitting a method

PRs welcome. A method submission is a directory under `methods/` containing:

- `method.md` — short description + theoretical justification
- `run.py` — deterministic script that takes the corpus + query set and emits ranked top-5 results per query
- `results.json` — precision@5, nDCG@5, blind-preference-rate (filled in after evaluation)

Open an issue first if you want to discuss baseline corpus or query set — the goal is a stable comparison point, not a moving target.

## Why this exists

Lubana et al. ([Goodfire AI, 2025–2026](https://www.youtube.com/watch?v=F9eEYWX64ZA)) named the tool gap: *"something like a SAE but which respects nonlinear geometry."* This is one falsifiable test of whether a geodesic-respecting retriever clears the bar that strong embeddings already set.
