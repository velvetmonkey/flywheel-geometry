# gold/

Reference labels for the benchmark. **Do not pass any of this to a method runner.**

## Files

- `targets.jsonl` — for each `query_id`, the hidden gold-standard targets:
  - `target_note_ids`: list of corpus note IDs that constitute correct retrieved items
  - `target_domains`: domains those targets span (used for cross-domain analysis)
  - `rationale`: free-text explanation of why these targets are the right answer

## Why this is split out from the public queries

An earlier file (`queries.jsonl`, since removed) carried the gold targets inside the same JSONL the runners consume — fields like `hidden_target_note_ids`, `hidden_domains`, and free-text rationale lived alongside the public query. Any future method submission could trivially read those fields and score itself perfectly. That defeats the benchmark.

By the convention used here:

- `queries.public.jsonl` is what every `methods/*/run.py` reads. It contains only `query_id`, `query_text`, and `is_control`.
- `gold/targets.jsonl` is what the **evaluator** reads after a run completes. It scores the run's `results/<method>-<date>.jsonl` against the gold targets, blind to the method.

Method runners that read this directory should be rejected at PR review.

## Control queries

3 of 30 queries are intra-domain controls (`is_control: true`). They retrieve from a single domain with no expected cross-domain bridge structure. They're scored *separately* — included in the dataset to provide a sanity floor, not part of the headline cross-domain metric.

| Query type | Count | Headline metric |
|---|---|---|
| Cross-domain bridge | 27 | yes |
| Intra-domain control | 3 | reported separately |
