# Synthetic Persona — Alex Reynold

The benchmark corpus is **entirely synthetic**. None of these notes describe real people, places, projects, or events. The persona below is a constructed device used to give the corpus internal coherence — every note in `notes/` is written as if Alex authored it.

This file exists so anyone running the benchmark can see the shape of the source material without confusing fiction for biography. If you publish a method against this corpus, please reproduce these notes verbatim — the cross-domain bridge structure depends on the consistent voice.

## Persona sketch

- **Name**: Alex Reynold
- **Location**: Brisbane, Queensland, Australia
- **Day job**: Quantitative analyst at *Meridian Capital* (a fictional multi-strategy hedge fund)
- **Partner**: Robin (a session musician, occasional mentions)
- **Horse**: Caspian — 8-year-old Andalusian gelding, dressage focus
- **Vehicle**: Polestar 2 (electric)
- **Side project**: *Crucible* — a personal task / decision tracking system
- **Mentor**: Marcus (former trader, occasional sounding board)
- **Coach**: Priya (riding coach)

## Cross-domain concept seeds

The corpus is designed so that the *same underlying concepts* surface in notes from different domains. Examples:

- **Feedback loops** — appear in equine training, AI activation steering, factor decay
- **Regime change / distribution shift** — quant finance, seasonal horse work, vehicle telemetry
- **Calibration** — risk models, reading the horse's energy, drone trim
- **Exploration vs exploitation** — groundwork training, R&D vs production, career fork
- **Compounding** — knowledge graph growth, fund returns, training progression
- **Tail dependence** — quant carry, weather contingencies on travel, equipment failure modes

A retrieval method that *only* uses cosine similarity over text embeddings should miss many of these bridges because the surface lexicon is completely different. A geodesic / manifold-aware method should surface them precisely because the underlying conceptual structure overlaps.

## Domain breakdown (10 × 5 = 50 notes)

| Prefix | Domain | Notes |
|---|---|---|
| `ai-` | AI / interpretability research | 5 |
| `eq-` | Equine / horse training | 5 |
| `qf-` | Quantitative finance | 5 |
| `ve-` | Vehicle / equipment ownership | 5 |
| `ph-` | Philosophy / personal musings | 5 |
| `tr-` | Travel / itinerary planning | 5 |
| `ca-` | Career / professional strategy | 5 |
| `sa-` | Software architecture | 5 |
| `ha-` | Home admin / operational | 5 |
| `dj-` | Daily journal | 5 |

## Authorship & licensing

All synthetic notes in this directory are part of the benchmark and inherit the repo's Apache-2.0 license. Do not treat any specific claim, metric, person, or place named here as real.
