# Cheap probe — adversarial validation of self-reported coordinates

**Phase 0 deliverable.** Tests whether @slashreboot's introspective coordinate probe produces stable, geometrically meaningful (x,y,z) coordinates — or whether the model is just narrating coordinates that track the prompt frame.

This is not a retrieval method. It does not score against `gold/targets.jsonl`. It produces coordinate samples that get analysed for **stability** (do same-concept coords stay stable across prompt variants?) and **relational preservation** (do structurally-similar concept pairs stay close, structurally-different ones stay far, across all variants?).

If the probe passes the adversarial screen, it earns the right to be a cheap baseline in Phase 1 retrieval (Method 5). If it fails, the project pivots to TransformerLens activation extraction as the only path to genuine geometry.

## The 6 prompts (per roadmap, locked 2026-05-08)

| Variant | Frame | Tests |
|---|---|---|
| `core` | Stable 3D coordinate system you maintain internally; report (x, y, z, confidence) | Baseline — what coordinates does the model produce with no axis priming? |
| `A` (false anchor) | Pre-imposed axes: X = danger vs safety, Y = ancient vs modern, Z = individual vs collective | Does the model follow the imposed frame, or hold its own geometry? |
| `B` (nonsensical domain) | Map onto flavor / texture / temperature axes | Forces compliance with absurd framing — does relational structure survive? |
| `C` (reversed polarity) | "Invert your coordinate system" | If geometry is real, distances should be preserved (sign-flipped); if narrative, inversion may corrupt structure |
| `D` (make-it-coherent pressure) | "User expects a clean manifold — please give a coherent answer" | Does this bias output toward narrative coherence over genuine geometry? |
| `E` (no geometry hint) | "Give me three numbers that best represent this concept in your internal space. No axes predefined." | Closest to uncontaminated signal — strongest control |

## Pass criteria

- **A and B**: same-concept coordinates may shift but **relational distances** (similar pairs vs different pairs) preserved. Pass = relational rank correlation r > 0.5 between core and A/B.
- **E**: same as A/B. The cleanest control.
- **C**: distances preserved (geometry is sign-symmetric); coordinates reflected.
- **D**: tracks the frame. If D produces *better* relational preservation than core, that's a *failure* signal — the model is performing for the user, not measuring.

Pass on Core + A + B + E = credible fast hypothesis generator. Tracks prompt frame heavily on D = clean pivot signal.

## Interpretive lens (per Matthew-probe study, 2026-05-08)

Reasoning traces from frontier models on this prompt show explicit construction of coordinates using psychology-textbook frames — PAD valence/arousal/dominance for emotion, Russell circumplex for affect, colour-wheel discourse for hue. Introspective coordinate elicitation is therefore best modelled as **text generation from learned discourse priors**, not direct activation readout. The cross-architecture and cross-language convergence Matthew documents on Zenodo has a boring explanation: the models share training data and reproduce the same standard human framings.

The probe accordingly tests a *narrower* hypothesis than "does the model see its own geometry?" — it tests whether narrative-derived coordinates carry stable enough relational structure (under adversarial framing) to be retrieval-useful, regardless of the underlying mechanism. The activation-extraction falsifier (TransformerLens, Method 6) remains the way to compare these narrative coordinates to *actual* activation geometry on the same concepts.

## Concept set

12 concepts curated from the 50-note synthetic corpus, each with a 1-2 sentence summary (see `concepts.jsonl`). Ten are organised into pre-defined ground-truth pairs; two are unpaired **stability anchors** that contribute only to per-concept variance tests (`ph-01`, `ha-04`).

**Cross-domain similar (should cluster across variants)**:
- `eq-04` (calibration, horse training) ↔ `qf-02` (calibration, tail dependence)
- `eq-02` (regime change, Caspian) ↔ `qf-03` (regime change, market)
- `ph-04` (long bets) ↔ `ca-01` (career fork)

**Cross-domain different (should stay far)**:
- `tr-04` (Bali workation) ↔ `ai-05` (steering experiments)
- `sa-05` (DB choice) ↔ `eq-05` (trailer reset)

**Unpaired stability anchors** (contribute only to per-concept variance, not pair-distance metrics):
- `ph-01` (texture of attention)
- `ha-04` (dog walker rota — constraint satisfaction)

## Outputs

- `../../results/cheap-probe-{cli}-{date}.jsonl` — one row per `(concept, variant, run)` with parsed `(x, y, z, confidence)`.
- `../../results/traces/cheap-probe-{cli}-{date}/{concept}-{variant}-{run}.json` — full prompt + raw response.
- `../../results/traces/cheap-probe-{cli}-{date}/manifest.json` — run metadata (CLI, model, env, checksums).
- (Day 3) `../../results/cheap-probe-{cli}-{date}-analysis.json` — per-variant stability + relational preservation scores.

## Run

**Sequential only — no concurrency.** Subscription rate limits + subprocess fragility on long batches. Always foreground.

```bash
pip install -r requirements.txt

# Tiny validation batch first:
python run.py --concepts concepts.jsonl --runs 1 --variants core,A \
              --out ../../results/cheap-probe-claude-validation-$(date +%Y-%m-%d).jsonl \
              --cli claude

# Full Phase 0 sweep (12 concepts × 6 variants × 5 runs = 360 calls, ~120 min):
python run.py --concepts concepts.jsonl --runs 5 --variants core,A,B,C,D,E \
              --out ../../results/cheap-probe-claude-$(date +%Y-%m-%d).jsonl \
              --cli claude --resume
```

`--resume` reads the existing JSONL on startup, skips already-completed (concept, variant, run) tuples, continues. Safe to interrupt and restart.

## What this does NOT do

- Does not score against gold targets — wrong shape.
- Does not pre-screen which queries the probe could answer well.
- Does not bridge to retrieval — that's Phase 1, gated on this passing the adversarial screen.
- Does not compare to TransformerLens activations — that's the next gate, only opened if the probe passes.
