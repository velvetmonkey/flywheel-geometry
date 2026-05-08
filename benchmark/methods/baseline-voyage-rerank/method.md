# Baseline: Voyage-3 + LLM rerank

Strong embedding-based retrieval baseline. The bar geodesic retrieval has to clear by ≥20% precision@5 to justify shipping.

## What this measures

Modern dense-embedding retrieval over the 50-note corpus, with a frontier LLM rerank pass on the top candidates. This is what `flywheel-memory` already ships variants of (BM25 + semantic via RRF). If geodesic doesn't beat this on cross-domain bridges, geodesic isn't ready.

## Inputs

- `../../corpus/notes.jsonl`
- `../../queries.jsonl`

## Outputs

- `../../results/voyage-rerank-{date}.jsonl` — for each query, top-5 reranked note IDs

## Run

```bash
export VOYAGE_API_KEY=...
export ANTHROPIC_API_KEY=...
pip install -r requirements.txt
python run.py --corpus ../../corpus/notes.jsonl --queries ../../queries.jsonl --out ../../results/voyage-rerank-$(date +%Y-%m-%d).jsonl
```

## Implementation

1. Embed each note with `voyage-3` (`voyageai` SDK).
2. Embed each query with the same model.
3. Cosine-rank → take top-15 candidates per query.
4. LLM rerank pass: send query + 15 candidate note titles+bodies to Claude (claude-sonnet-4-6) with a "rank these by relevance to the query" prompt; take top-5.
5. Write top-5 to JSONL.

Caching: embeddings cached by note ID + model version to avoid re-embedding on repeat runs.
