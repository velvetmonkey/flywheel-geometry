# Changelog

## [Unreleased]

### Added
- Initial product vision README
- Apache-2.0 license
- Project scaffolding
- **2026-05-08** — 50-note synthetic corpus + 30 cross-domain queries (27 primary + 3 control), gold/public split with hidden targets at `benchmark/gold/targets.jsonl`
- **2026-05-08** — Six baseline method directories scaffolded under `benchmark/methods/` (BM25, Voyage native rerank, Voyage rerank, Voyage rationale rerank, Voyage rationale Claude rerank, Voyage frontier); Method 6 (geodesic, activation-extraction) pre-registered at `benchmark/methods/method-geodesic/` for v0.2
- **2026-05-08** — Subscription-CLI dispatch (`claude` / `codex` / `gemini`) for LLM-rerank baselines, replacing API-key paths so the v0.1 baselines run on the user's existing subscription
- **2026-05-08** — `benchmark/eval.py` with five pre-scoring guards (row count, unique query_ids, unique top-K IDs, all retrieved IDs in corpus, gold-leak tripwire) and per-query sidecars; running leaderboard at `benchmark/results/RESULTS.md` with v0.1 tie marker (±0.05 absolute p@5) documented as a debug aid distinct from the v0.2 ≥20% lift ship criterion
- **2026-05-08** — BM25 sanity floor scored on the 27 primary queries (p@5 0.252) — corpus is partially keyword-contaminated; bridge-concept names like *calibration*, *regime change*, *feedback loops* appear as literal strings across domain notes. Finding logged on the leaderboard; v0.2 corpus-authoring lesson is to push more queries toward the hard-bridge end where the gold targets share underlying structure but not surface vocabulary.
- **2026-05-08** — Cheap-probe Phase 0 method scaffolded at `benchmark/methods/cheap-probe/` — six prompt variants (core baseline, A false-anchor, B nonsensical-domain, C inverted-polarity, D coherence-pressure, E no-axes), 12 concepts curated from the synthetic corpus (10 organised into five ground-truth pairs + 2 unpaired stability anchors), `analyze.py` computing per-(concept, variant) coordinate stability and Spearman rank correlation on pair-distance vectors. 18-call validation batch on Claude Opus 4.7 cleared zero parse failures.
- **2026-05-08** — README hygiene pass: gold-firewall surfaced under "How we'll know this works", assumption #2 wording updated to reference the actual baseline names on the 27 primary queries, Status section now reflects current scaffold + leaderboard + sweep state.

### Changed
- **2026-05-08** — README assumption #1 refined to track the [Matthew-probe study](https://doi.org/10.5281/ZENODO.18176077) finding: introspective coordinate elicitation is text generation from learned discourse priors (PAD axes, Russell circumplex), not direct activation readout. The remaining narrower hypothesis — whether those narrative-derived coords correlate enough with activation-derived geometry to be retrieval-useful — is what the cheap-probe + Method 6 comparison answers.
- **2026-05-08** — Method 6 (geodesic) base model and layer locked: Llama-3.1-8B-Instruct (primary) / Gemma-2-9B (replication), single residual-stream layer in the 8–12 band with layer 16 reserved as strong secondary and layer 0 as syntactic-control sanity check. Replaces the prior "Llama 3 70B or Gemma 3 27B, mid-network layer" placeholder. Smaller model chosen so Phase 0 + Phase 1 plumbing runs on consumer hardware; larger-model robustness check deferred to v0.3+.

### Removed
- CTMU/Langan reference from `docs/philosophy.md` (kept off the public surface; cosmological framings remain in philosophy.md only)

*Started 2026-05-07.*
