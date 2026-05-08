# geodesic-bench leaderboard

Public running results for [`benchmark/`](../README.md).

## Decision criterion

**Method 6** (activation-derived geodesic retrieval, v0.2) ships only if it shows ≥20% precision@5 lift over **all three of Method 3, Method 4a, and Method 4b** on the **27 cross-domain bridge queries** (primary metric). Control queries are reported separately, never mixed into the headline.

If Method 6 ties or fails to clear that bar against any of those three baselines, the project pivots — either to bridge-tension via relational structure, to direct activation extraction without the introspective wrapper, or away from the manifold thesis entirely. The pivot post is the launch.

### v0.1 tie marker (debug aid only)

For v0.1 harness validation, the eval script flags any method whose primary p@5 is within **0.05 absolute** of the best as `tied_with_best: yes`. **This is a debugging aid, not the v0.2 ship criterion.** The real criterion remains the ≥20% lift above. Use the v0.1 tie marker to surface methods that are statistically indistinguishable at this small N (27 primary queries); use the ≥20% bar to make ship/pivot decisions.

## Leaderboard

Each row records one scored run. Methods listed in dependency order (sanity floor → strong baselines → kill-product → hypothesis). The eval script appends or replaces rows by method name on every invocation.

| Method | P@5 (primary 27) | nDCG@5 (primary) | P@5 (control 3) | nDCG@5 (control) | Run date |
|---|---|---|---|---|---|
<!-- leaderboard:start -->
| `bm25-2026-05-08` | 0.252 (24/27) | 0.439 | 0.267 (2/3) | 0.421 | 2026-05-08 |
<!-- leaderboard:end -->

Methods 1–4b are scaffolded under [`methods/`](../methods/); Method 6 (geodesic) is pre-registered at [`methods/method-geodesic/`](../methods/method-geodesic/) and lands in v0.2. Each row above is one scored run; the eval script appends or replaces by method name on each invocation.

### v0.1 surprise: BM25 is not at zero

Pre-run expectation: cross-domain bridge queries should be lexically disjoint by design, so BM25 primary p@5 should be near zero.

Observed: BM25 primary p@5 = 0.252 (24 of 27 primary queries had ≥1 hit in top-5). Inspection of the synthetic corpus shows that the *bridge-concept names themselves* — "calibration," "feedback loops," "regime change," "tail dependence," "exploration vs exploitation" — appear as literal strings in multiple domain notes. BM25 picks those up. The corpus is **partially keyword-contaminated**.

The 3 cross-domain queries BM25 missed entirely (q23, q29, q30) are the *cleanest* bridge tests — gold targets that share underlying concept structure but not surface vocabulary. Method 6 (geodesic) needs to specifically rescue those.

**v0.2 corpus authoring lesson**: aim more queries toward the hard-bridge end (gold targets with no lexical overlap to the query) rather than the soft end. The current 27 primary queries are graded; v0.2 should be more uniform-hard.

## How rows get here

```bash
cd benchmark
python eval.py --gold gold/targets.jsonl --queries queries.public.jsonl \
  results/<method-and-date>.jsonl [results/<method2>...]
```

The evaluator runs five pre-scoring guards on each results file (row count, unique query_ids, unique top-K IDs, all retrieved IDs in corpus, no gold-leak fields). Any failure exits non-zero and **does not** append a row — garbage runs do not pollute the leaderboard.

Per-method per-query traces with `hits_at_5` boolean arrays land at `results/<method>-<date>-eval.json` for downstream debugging.
