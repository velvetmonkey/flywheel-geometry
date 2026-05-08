# Frontier baseline: voyage-4-large + rerank-2.5

Same shape as Method 2 (`baseline-voyage-native-rerank`), but pinned to Voyage's *current* frontier embedding + reranker rather than the stable-comparison `voyage-3` + `rerank-2`. Provides a "are we behind the latest commercial offering?" signal.

## Why two voyage baselines

The benchmark's stable comparison anchors are Methods 1–4b — all pinned to `voyage-3` and `rerank-2`. Those numbers should be reproducible by anyone running the benchmark today, three months from now, or three years from now. They define what each method clears.

Voyage rolls models forward — `voyage-4-large` and `rerank-2.5` are the current best at the time of writing, but Voyage will publish newer ones. This baseline tracks the moving target. It's *not* a stable comparison anchor; results drift as Voyage upgrades. Worth running, worth reporting separately, but not part of the headline kill-product test.

## What this measures

If Method 6 (geodesic) only beats `voyage-3 + rerank-2` and not `voyage-4-large + rerank-2.5`, the geodesic claim weakens — the gain may just be "old commercial stack vs new commercial stack." If Method 6 beats *both*, the claim is more robust.

## Inputs

- `../../corpus/notes.jsonl`
- `../../queries.public.jsonl`

## Outputs

- `../../results/voyage-frontier-{date}.jsonl`
- `../../results/traces/voyage-frontier-{date}/q<NN>.json`
- `../../results/traces/voyage-frontier-{date}/manifest.json` — captures the *exact* Voyage model versions hit at run time

## Run

```bash
export VOYAGE_API_KEY=...
pip install -r requirements.txt
python run.py --corpus ../../corpus/notes.jsonl --queries ../../queries.public.jsonl \
              --out ../../results/voyage-frontier-$(date +%Y-%m-%d).jsonl
```

Optional flags:
- `--embed-model voyage-4-large` (default) / `voyage-4` / `voyage-4-lite`
- `--rerank-model rerank-2.5` (default) / `rerank-2`

## Implementation

Identical to Method 2, but with different model strings. Embeddings cache key includes `--embed-model` so the cache stays correct when models are swapped.
