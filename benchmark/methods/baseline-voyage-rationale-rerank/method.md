# Baseline: Voyage-3 + LLM rerank with bridge-rationale generation

**The kill-product test.** Codex's round-2 #6 critique elevated to a tracked baseline.

## What this measures

Same retrieval pipeline as `baseline-voyage-rerank`, but at rerank time the LLM is *also* asked to generate a bridge rationale — a short paragraph explaining how each candidate connects to the query — *before* the final ranking.

If raters prefer this baseline's results at equal or higher rates than plain rerank, the manifold "effect" geodesic retrieval claims to surface is **the LLM explaining adjacency, not the geometry surfacing it**. The product premise dies.

If geodesic retrieval can't beat *both* plain rerank AND rationale-augmented rerank by ≥20% precision@5, the project pivots — either to bridge-tension via relational structure (Codex's reframe) or away from the manifold thesis entirely.

## Inputs

- `../../corpus/notes.jsonl`
- `../../queries.jsonl`

## Outputs

- `../../results/voyage-rationale-rerank-{date}.jsonl` — top-5 reranked + per-candidate rationales

## Run

```bash
export VOYAGE_API_KEY=...
export ANTHROPIC_API_KEY=...
pip install -r requirements.txt
python run.py --corpus ../../corpus/notes.jsonl --queries ../../queries.jsonl --out ../../results/voyage-rationale-rerank-$(date +%Y-%m-%d).jsonl
```

## Implementation

1. Embed corpus + queries with `voyage-3`.
2. Cosine-rank → top-15 candidates per query.
3. For each candidate, ask Claude to generate a 1-paragraph bridge rationale: "How does this note connect to the query? What's the cross-domain insight?"
4. Append rationale to candidate text, re-embed candidate with `voyage-3` *with rationale appended*.
5. Re-rank by cosine similarity to query embedding.
6. Take top-5.

This is deliberately strong — it gives the LLM full opportunity to manufacture a rationale that *makes* the candidate look relevant. The point is to measure whether the rationale-generation step alone gets us most of the cross-domain bridge effect that the manifold claim is supposed to deliver.
