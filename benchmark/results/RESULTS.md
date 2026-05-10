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
| `cli-direct-rerank-claude-sonnet-2026-05-08` | 0.370 (26/27) | 0.670 | 0.267 (2/3) | 0.490 | 2026-05-08 |
| `voyage-native-rerank-2026-05-08` | 0.333 (26/27) | 0.624 | 0.400 (2/3) | 0.628 | 2026-05-08 |
| `method-geodesic-llama31-8b-trial2-layer10-2026-05-10` | 0.104 (11/27) | 0.154 | 0.000 (0/3) | 0.000 | 2026-05-10 |
| `method-geodesic-llama31-8b-trial2-layer10-randomctrl-2026-05-10` | 0.089 (12/27) | 0.127 | 0.000 (0/3) | 0.000 | 2026-05-10 |
| `method-geodesic-llama31-8b-trial2-layer12-2026-05-10` | 0.074 (10/27) | 0.105 | 0.067 (1/3) | 0.210 | 2026-05-10 |
| `method-geodesic-llama31-8b-trial2-layer8-2026-05-10` | 0.059 (8/27) | 0.089 | 0.000 (0/3) | 0.000 | 2026-05-10 |
<!-- leaderboard:end -->

Methods 1–4b are scaffolded under [`methods/`](../methods/); Method 6 (geodesic) is pre-registered at [`methods/method-geodesic/`](../methods/method-geodesic/) and lands in v0.2. Each row above is one scored run; the eval script appends or replaces by method name on each invocation.

### v0.1 finding — cheap-probe falsified its pre-registered adversarial screen on Sonnet 4.6

Phase 0 ran 360 calls (12 concepts × 6 prompt variants × 5 runs) of [@slashreboot's](https://x.com/slashreboot) introspective coordinate probe on `claude-sonnet-4-6`, with the pre-registered adversarial-framing battery from [`methods/cheap-probe/method.md`](../methods/cheap-probe/method.md).

Pass criterion was Spearman rank correlation `ρ > 0.5` between pair distances under `core` and each adversarial variant. **All four screens failed** (A: 0.078; B: 0.290; C: 0.159; E: 0.108). Variant D — pre-registered as the failure-signal trap — fired on cue: D produced 18× the ground-truth pair separation of `core` (5.16 vs 0.27) only when the prompt explicitly told the model "the user expects a coherent manifold." The model performs coherence under that framing; it does not measure it.

Sonnet refused 43 / 360 calls outright (mostly variant C: *"No access to my own activation space — any numbers I output would be fabricated"*). Earlier in the day, Claude Haiku 4.5 refused 5 / 5 on `core`; the run was switched to Sonnet for that reason.

Per [`method.md`](../methods/cheap-probe/method.md): *"If it fails, the project pivots to TransformerLens activation extraction as the only path to genuine geometry."* That pivot is now active. Full narrative + data links: [`docs/v0.1-pivot.md`](../../docs/v0.1-pivot.md).

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
