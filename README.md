<div align="center">
  <img src="header.png" alt="Flywheel" width="256"/>
  <h1>Flywheel Geometry</h1>
  <p><strong>An open, pre-registered study of cross-domain knowledge retrieval.</strong></p>
  <p><em>Trial 2 resolved 2026-05-10: <strong>FAIL</strong>. The pre-registered falsifier ran on the locked benchmark and lost to all three baselines including BM25. Full audit: <a href="./docs/trial2-postmortem.md">docs/trial2-postmortem.md</a>.</em></p>
</div>

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Pre-registered Study: Trial 2 FAIL](https://img.shields.io/badge/Pre--registered_Study-Trial_2_FAIL-red)](#falsification-log)

Open question: do cross-domain bridges in a personal knowledge vault — notes that share underlying structure but no surface vocabulary — surface more reliably via activation-derived geometry than via strong embedding retrieval? Recent interpretability work suggests neural networks encode meaning on curved manifolds where geodesic distance follows the curve while cosine cuts through the void ([Lubana et al., 2025–2026](https://www.youtube.com/watch?v=F9eEYWX64ZA); [@slashreboot, 2026](https://doi.org/10.5281/ZENODO.18176077)) — this repo turns that into a falsifiable benchmark with locked baselines, hidden gold targets, and pre-registered ≥20 % p@5 lift criteria. See Falsification Log below.

---

## Falsification log

Each trial is pre-registered before any data lands. Resolution is one of `falsified`, `validated`, `running`, or `deferred` — the same vocabulary as `flywheel-ideas` outcome states. Public commitments only; the private vault holds method-design notes that don't bind outside readers.

| # | Trial | Pre-registered | Status | Numbers / outcome |
|---|---|---|---|---|
| 0 | Kill-product floor — pinned-model retrieval baselines on 27 cross-domain bridge queries | 2026-05-08 (eval guards + corpus + queries locked at commit `c023a94`) | ✅ **scored** (3 of 4 rows) | BM25 0.252 · Voyage-native 0.333 · CLI direct rerank (Sonnet 4.6) **0.370** primary p@5. Method 4a / 4b held until Method 6 spike validates the path. |
| 1 | Cheap-probe Phase 0 — does @slashreboot's introspective coordinate elicitation produce stable, retrieval-useful relational structure under adversarial framing? | 2026-05-08 (variants A–E + decision rules locked at commit `f529c0b`) | ❌ **falsified 2026-05-08** | All four primary screens failed (ρ A 0.078 · B 0.290 · C 0.159 · E 0.108 — pre-reg bar ρ > 0.5); variant-D failure-signal trap fired (D ground-truth pair separation 5.16 vs core's 0.27, 18×). Full data + narrative: [`docs/v0.1-pivot.md`](./docs/v0.1-pivot.md). Refutes flywheel-ideas `asm-HvE9muhM` + `asm-VotY4n8g`; outcome `out-MyLPFpg7`. |
| 2 | Method 6 — kNN + geodesic distance over residual-stream activations at layers 8 / 10 / 12 of `meta-llama/Llama-3.1-8B-Instruct` beats the kill-product floor on the 27 primary queries by ≥20 % p@5 | 2026-05-08 (primary layer 10 + sensitivity 8/12 + random-kNN negative control + activation contract locked at commit `82540a3`) | ❌ **falsified 2026-05-10** | Layer-10 (primary) **p@5 = 0.104 (11/27)** vs required ≥0.444. Lost to BM25 (0.252, −59%), voyage-native-rerank (0.333, −69%), and the kill-product floor cli-direct-rerank-claude-sonnet (0.370, −72%). Within-method random-kNN control at layer 10 scored 0.089; geodesic adds only +0.015 absolute, statistically tied at n=27 (eval flagged `tied_with_best`). Layer sensitivity: layer-12 0.074, layer-8 0.059. Activations SHA256 `bf619cf5151…`. Full audit: [`docs/trial2-postmortem.md`](./docs/trial2-postmortem.md). Public announcement: [@thevelvetmonke thread](https://x.com/thevelvetmonke/status/2053522315162607659) (6 posts + 1 reply). Refutes flywheel-ideas `asm-3zmj1VGB`; outcome record assignment in flywheel-ideas pending. |
| 3 | Method 6 vs rationale-augmented baselines — does the manifold effect survive the kill-product floor's full set (Method 3 + 4a + 4b)? | Triggers only on Trial 2 validate branch | 🚫 **deferred indefinitely** (Trial 2 falsified) | This branch was conditional on Trial 2 PASS. Method 4a / 4b full runs no longer triggered. Locked assumption `asm-wxSuxhBk` deferred indefinitely. |

The artifact-vs-geometry distinction relevant to Trial 1: the cheap-probe sweep replicated @slashreboot's *surface artifact* (different LLMs output convergent self-reported coordinates on identical prompts) but **not** the underlying claim that those coordinates measure activation geometry. Reasoning traces show models constructing coordinates from PAD valence/arousal axes + Russell-circumplex priors learned in training. Method 6 (Trial 2) is the path that would read activation manifolds directly — which is why the project pivoted there rather than discarding the manifold thesis on Trial 1's failure.

The full leaderboard with all per-query metrics is at [`benchmark/results/RESULTS.md`](./benchmark/results/RESULTS.md). The flywheel-ideas decision ledger (idea `idea-b4ZeRCoa`) holds the full assumption / outcome state.

---

## How this slots in

Flywheel Geometry is an **extension library for [flywheel-memory](https://github.com/velvetmonkey/flywheel-memory)** — the local-first MCP server that turns an Obsidian vault into safe AI memory. Today flywheel-memory ships hybrid BM25 + semantic search via Reciprocal Rank Fusion. Flywheel Geometry adds a *bridge-finder* axis: a separate query mode answering "what's structurally adjacent across domains?" alongside the existing precision-search modes. If the empirical experiment lands, the new axis ships as an optional layer — same MCP surface, additional `mode: "bridge"` discriminator, no replacement of existing search.

Part of the [Flywheel](https://github.com/velvetmonkey) suite:

- **[flywheel-memory](https://github.com/velvetmonkey/flywheel-memory)** — local-first MCP server. Hybrid BM25 + semantic search, knowledge graph, safe writes over an Obsidian vault.
- **[flywheel-crank](https://github.com/velvetmonkey/flywheel-crank)** — Obsidian plugin. Visual layer over Memory's graph: sidebar, vault health, semantic search UI.
- **[flywheel-ideas](https://github.com/velvetmonkey/flywheel-ideas)** — falsifiable decision ledger. Pre-registered assumptions, multi-model AI council dissent, outcome-driven refutation propagation.
- **flywheel-geometry** *(this repo)* — geodesic retrieval extension. Pre-registered study of cross-domain bridge-finding via activation manifolds.
- **[flywheel-concept](https://github.com/velvetmonkey/flywheel-concept)** — research programme on whether cross-model activations reveal structured concept geometry.

---

## The Problem

Standard retrieval measures cosine similarity between embedding vectors — linear distance in high-dimensional space. But concepts aren't encoded linearly. Neural networks encode meaning as curved manifolds: hue wheels, temporal spirals, helices, emotional circumplexes.

Steering linearly between two concepts cuts through the void where no valid meaning exists. The model becomes incoherent. Cosine similarity does the same to retrieval — it measures distance through regions that don't correspond to real semantic relationships.

Sparse autoencoders (SAEs) tile the manifold into fragments. They shatter the helix into disconnected shards. You get pieces of the shape, not the shape itself.

---

## What this is (and isn't)

**Is**: an open benchmark + reference implementation testing whether activation-derived geometry surfaces useful cross-domain bridges that strong embedding retrieval misses.

**Isn't**: a claim that activation geometry beats cosine for general retrieval. It probably doesn't, and we're not testing that. Use the right tool for each question — cosine for precision search, geodesic for bridge-finding.

The bet: notes from different domains — horse training, AI architecture, finance, philosophy — that share underlying structure (feedback loops, regime change, calibration, exploration vs exploitation) should surface together via geodesic adjacency on the activation manifold. No shared keywords. No explicit links. Just structural proximity.

Whether activation-derived geometry actually delivers this signal more reliably than rationale-augmented embedding retrieval is the open empirical question.

---

## How It Works

**Primary method (the real hypothesis)**:
Extract residual-stream activations from a single open-source model (Llama 3 / Gemma 3) at a pre-registered layer, build a kNN graph over note activations weighted by activation-space distance, retrieve via shortest-path or diffusion distance. No magic 3D projection in the mainline; UMAP is for visualisation only, not the retrieval substrate.

**Cheap baseline (the introspective probe, not measurement-grade)**:
[@slashreboot](https://x.com/slashreboot)'s zero-shot coordinate elicitation — a single user message asks a frontier model to introspect and return (x,y,z) coordinates. We treat this as a *cheap hypothesis generator* for what geometry might exist, **not** as a measurement of activation structure. We will test whether self-reported coordinates track activation-derived structure under adversarial conditions (false anchors, fake coordinate frames, synthetic concept domains). That adversarial comparison is milestone 1, not a later safety check.

**Causal filter (post-baseline)**:
If the basic geodesic method beats the rationale-augmented baseline, candidate subspaces get filtered through DAS (Distributed Alignment Search; Geiger et al.) for causal efficacy — keeping only directions that actually drive behaviour, not just correlate with it. Phase 2 work, gated on Phase 1 success.

---

## How we'll know this works

We're not assuming geodesic retrieval surfaces better cross-domain bridges than rationale-augmented embedding search — we're testing it. v0.1 plumbing runs on a 50-note synthetic corpus to validate the harness; v0.2 scales to a 500-note corpus before any claim. On 30 hand-written cross-domain bridge queries with hidden gold targets, blind-rated top-5 results need to show ≥20% precision gain over **two** baselines:

1. Voyage-3 + LLM rerank.
2. Voyage-3 + LLM rerank where the LLM is allowed to generate bridge rationales for each candidate.

The second baseline is the one that distinguishes retrieval from presentation. If raters prefer rationale-augmented embeddings at equal or higher rates, the manifold "effect" is the LLM explaining adjacency, not the geometry surfacing it.

**Gold firewall.** Method runners consume only [`benchmark/queries.public.jsonl`](./benchmark/queries.public.jsonl); the hidden targets in [`benchmark/gold/`](./benchmark/gold/) are read exclusively by the evaluator. The eval harness rejects any results file that includes gold-shaped fields (`target_note_ids`, `target_domains`, `rationale`) — so a runner that accidentally peeks fails the pre-scoring guard and never lands a row on the leaderboard.

Results — confirming or refuting — get published here.

A separate falsifier targets the introspective probe itself: extract activations via TransformerLens for the same concepts, compare to self-reported coordinates under adversarial controls (false anchors, fake coordinate frames, synthetic concept domains). If self-report tracks activation-derived relational structure, the probe is measurement-grade. If it tracks the prompt's framing instead, the project pivots to direct activation extraction.

### Tracked through [flywheel-ideas](https://github.com/velvetmonkey/flywheel-ideas)

The project's central bets are registered as falsifiable assumptions in the [flywheel-ideas](https://github.com/velvetmonkey/flywheel-ideas) decision ledger — the sibling project built for exactly this shape of work. Each assumption carries a falsifier and a resolution criterion; multi-model AI council dissent is logged at registration; outcomes (confirm / refute) propagate to dependent claims when experiments resolve.

The currently tracked assumptions, in their flywheel-ideas state as of 2026-05-08 (idea `idea-b4ZeRCoa`):

1. **`asm-HvE9muhM` — Self-reported (x,y,z) coordinates carry retrieval-useful relational structure.** Status: ❌ **REFUTED 2026-05-08** by outcome [`out-MyLPFpg7`](./docs/v0.1-pivot.md). Even though the [Zenodo reasoning traces](https://doi.org/10.5281/ZENODO.18176077) show the mechanism is text generation from learned discourse priors (PAD valence/arousal, Russell circumplex, colour-wheel framings), the narrower bet — that the coords still carried enough relational structure under adversarial framing to be retrieval-useful — also failed. Cheap-probe Phase 0 on Sonnet 4.6 produced rank correlations of 0.078 / 0.290 / 0.108 against core for variants A / B / E (all below the pre-registered 0.5 bar) and the variant-D failure-signal trap fired. Falsifier (a) refutes; falsifier (b) — TransformerLens correlation — superseded by the project's pivot to direct activation extraction (Method 6).
2. **`asm-3zmj1VGB` — Activation-derived geometry contains retrieval-useful structure that strong embeddings miss.** Status: ❌ **REFUTED 2026-05-10** by the Trial 2 outcome (full audit: [`docs/trial2-postmortem.md`](./docs/trial2-postmortem.md)). The central bet of v0.1 — that residual-stream kNN + geodesic distance would surface cross-domain bridges that strong embedding retrieval missed — failed cleanly: layer-10 p@5 = 0.104 vs required ≥0.444 (~4× short), beaten by BM25 (0.252), voyage-native-rerank (0.333), and the LLM-rerank kill-product floor (0.370). Within-method random-kNN at layer 10 was 0.089 — geodesic added only +0.015 absolute, statistically indistinguishable. Outcome record assignment in flywheel-ideas pending.
3. **`asm-VotY4n8g` — Coordinate stability across runs is measurement-grade**, not stable narrative priors. Status: ❌ **REFUTED 2026-05-08** by the same outcome [`out-MyLPFpg7`](./docs/v0.1-pivot.md). Variant D produced 18× the ground-truth pair separation of `core` only when the prompt explicitly told the model to give a coherent answer — within-variant stability turned out to reflect deterministic narrative generation rather than geometric measurement.
4. **`asm-7oiyCMji` — Manifold proximity outperforms bridge-tension** (high embedding distance × high relational similarity) on cross-domain bridges. Status: ⏸️ open, **not load-bearing** for v0.1; tested only if Method 6 lands a positive result and bridge-tension becomes a real alternative product framing.
5. **`asm-wxSuxhBk` — Human-rated bridge value is not explained by generic embedding similarity + LLM rationale generation.** Status: 🚫 **DEFERRED INDEFINITELY** post-Trial-2. This assumption was joint-resolved-with-`asm-3zmj1VGB`'s validate branch (Method 4a + 4b full runs were conditional on Trial 2 PASS). Trial 2 failed; Method 4a / 4b full runs no longer trigger; this assumption is no longer load-bearing for v0.1.

The Anti-Portfolio post-mortem memo for the cheap-probe refutation (`out-MyLPFpg7`) is in the flywheel-ideas vault — root cause, what we expected, what actually happened, and the lesson on pre-registering coherence-pressure variants for any future probe that depends on a model self-reporting structure.

**Update 2026-05-10:** `asm-3zmj1VGB` refuted. The project has pivoted. The pivot is the launch. Public FAIL announcement thread: https://x.com/thevelvetmonke/status/2053522315162607659. Repo postmortem: [`docs/trial2-postmortem.md`](./docs/trial2-postmortem.md). Sibling [`flywheel-concept`](https://github.com/velvetmonkey/flywheel-concept) becomes the primary research lane under a separately pre-registered claim. Watching the thesis fail in the open was the strongest brand outcome the bet could produce — that outcome is now in the record.

---

## Theoretical Foundation

- **Cross-architecture convergence on self-reported coordinates** — Gemma, Llama, and GPT independently produce similar `(x, y, z)` outputs when prompted to introspect on the same concept (hue wheels, temporal helices, emotional circumplexes). *([Matthew (@slashreboot)](https://x.com/slashreboot), *Zero-Shot Geometric Probing Reveals Universal Cognitive Manifolds in LLMs*, Jan 2026 — [doi:10.5281/ZENODO.18176077](https://doi.org/10.5281/ZENODO.18176077))*. **What this project's 2026-05-08 cheap-probe sweep replicated and what it did NOT**: we reproduced the *surface artifact* — different models output convergent coordinates on identical prompts — but the published reasoning traces (and our adversarial sweep, see [`docs/v0.1-pivot.md`](./docs/v0.1-pivot.md)) show the mechanism is **shared training-data priors** (PAD valence/arousal, Russell circumplex, colour-wheel framings from psychology textbooks), not shared internal geometry. The convergence is real; the interpretation that it reads activation manifolds is not. Method 6 (residual-stream kNN via TransformerLens) is the path that would read activation manifolds directly.

- **Cross-modality convergence on physical reality** — Edamadaka, Yang, Li, Gómez-Bombarelli (MIT), *Universally Converging Representations of Matter Across Scientific Foundation Models* — [arXiv:2512.03750](https://arxiv.org/abs/2512.03750) (Dec 2025). ~60 foundation models across string-, graph-, 3D-atomistic, protein, and LLM architectures converge on shared representations of matter without coordinated training. Better-performing models converge more strongly; weaker ones scatter to architecture-specific manifolds. The strongest available evidence that the geometry is a property of the territory, not the map.

- **Platonic Representation Hypothesis** — Huh, Cheung, Wang, Isola, *The Platonic Representation Hypothesis* — [arXiv:2405.07987](https://arxiv.org/abs/2405.07987) (ICML 2024). Different model families converge toward a shared statistical model of reality `Z` as scale, data volume, and task diversity grow; cross-modal alignment increases with capability. The unifying claim this project applies to retrieval.

- **Geometry = behavior** — Representation geometry is a direct reflection of data statistics and model beliefs. To control behavior you must respect the geometry. Geodesic paths stay on the manifold; linear paths enter the void. *([Ekdeep Singh Lubana (@EkdeepL)](https://x.com/EkdeepL) et al., [Goodfire AI (@GoodfireAI)](https://x.com/GoodfireAI), 2025–2026 — [talk](https://www.youtube.com/watch?v=F9eEYWX64ZA))*

- **Marr's three levels** — Behavior, algorithms, and representations are reflections of each other because the model learned the world's distribution. *(David Marr, *Vision*, 1982)*

- **The tool gap** — *"We will probably need tools which can capture these geometries in a general fashion — something like a SAE but which respects nonlinear geometry."* *(Ekdeep Singh Lubana, Goodfire AI, 2026)*

---

## Key References

**Activation geometry & SAE critique** (the methodological foundation):
- Hindupur, Lubana, Fel, Ba — *[Projecting Assumptions: The Duality Between Sparse Autoencoders and Concept Geometry](https://arxiv.org/abs/2503.01822)* (NeurIPS 2025) — SAEs impose structural priors that determine what concepts can be detected. Direct overlap with what this benchmark targets.
- Bigelow, Wurgaft, Wang, Goodman, Ullman, Tanaka, Lubana — *[Belief Dynamics Reveal the Dual Nature of In-Context Learning and Activation Steering](https://arxiv.org/abs/2511.00617)* (Nov 2025) — context modulation and activation steering are mathematically dual. Justifies treating activation-space interventions as causally load-bearing.
- Ekdeep Singh Lubana et al. — nonlinear geometry + geodesic steering paper series with Thomas Fel, Goodfire AI (upcoming 2026; "brace for shapes").

**Causal abstraction (Phase 2 dependency)**:
- Atticus Geiger et al. — Distributed Alignment Search (DAS), causal-abstraction framework. Used downstream to filter spurious geometry once basic retrieval lands.

**Cheap baseline source**:
- Matthew Steiniger (@slashreboot) — *Zero-Shot Geometric Probing Reveals Universal Cognitive Manifolds in Large Language Models* — [doi:10.5281/ZENODO.18176077](https://doi.org/10.5281/ZENODO.18176077) (Jan 2026). The introspective probe used as a baseline here. We treat this as a cheap hypothesis generator, not as activation measurement.

**Probe-based supervision (relevant lineage, not direct evidence for geodesic retrieval)**:
- Prasad, Watts, Merullo, Gala, Lewis, McGrath, Lubana — *[Features as Rewards: Scalable Supervision for Open-Ended Tasks via Interpretability](https://arxiv.org/abs/2602.10067)* (Feb 2026) — RLFR cuts hallucination by 58% via probe-based confidence. Demonstrates that activation probes can become supervision signals. *Not* evidence that geodesic retrieval beats cosine; cited for the broader methodological lineage.

**Background**:
- Goodfire AI — *[The World Inside Neural Networks](https://www.goodfire.ai/research/the-world-inside-neural-networks)* (2025) — accessible essay introducing the geometry-zoo framing.
- David Marr — *Vision* (1982) — three-levels-of-analysis frame; useful philosophical scaffold rather than direct citation.

---

## Credit & Collaboration

This work builds on, does not extend, the underlying interpretability research. We are applying findings from Goodfire AI ([@EkdeepL](https://x.com/EkdeepL), [@GoodfireAI](https://x.com/GoodfireAI), [@thomas_fel_](https://x.com/thomas_fel_)) and [Matthew (@slashreboot)](https://x.com/slashreboot) to personal knowledge retrieval. Coauthorship on derivative academic work belongs upstream.

---

## Status

Live trial state — including pinned numbers for the three scored kill-product-floor rows and the open Method 6 gate — lives in the [Falsification Log](#falsification-log) at the top of this README. The full leaderboard with per-query metrics is at [`benchmark/results/RESULTS.md`](./benchmark/results/RESULTS.md).

**30-day milestone (was 2026-06-07).** Both criteria resolved early:
1. ✅ Kill-product floor scored on the 27 primary queries — three of four rows landed (BM25 0.252, voyage-native-rerank 0.333, cli-direct-rerank-claude-sonnet 0.370). Method 4a / 4b full runs **deferred indefinitely** post-Trial-2 (they were conditional on the validate branch).
2. ✅ Cheap-probe Phase 0 resolved — **falsified 2026-05-08** along the variant-D failure-signal trap (full data: [`docs/v0.1-pivot.md`](./docs/v0.1-pivot.md)).
3. ✅ **Method 6 (Trial 2) resolved — falsified 2026-05-10** (full audit: [`docs/trial2-postmortem.md`](./docs/trial2-postmortem.md)). Layer-10 p@5 = 0.104 vs required ≥0.444. The central v0.1 bet `asm-3zmj1VGB` is REFUTED.

**The pivot is the launch.** The public FAIL announcement thread (https://x.com/thevelvetmonke/status/2053522315162607659) and the repo postmortem are the v0.1 deliverables. Watching the thesis fail in the open with full pre-registration discipline was the strongest brand outcome the bet could produce. That outcome is now in the record.

The `flywheel-geometry` repo is **closed at v0.1** with the falsifier ran-and-failed outcome as the canonical artifact. Future work on cross-model concept geometry continues at sibling project [`flywheel-concept`](https://github.com/velvetmonkey/flywheel-concept) under a separately pre-registered claim.

### Goodfire Ember note (2026-05-08)

Goodfire's hosted [Ember](https://goodfire.ai) interpretability platform was investigated as a path for Method 6's activation extraction; the SDK exposes **SAE-decoded feature activations**, not raw residual-stream tensors, and self-serve API access was deprecated Feb 2026 in favour of select-partner onboarding. Method 6 as locked (residual-stream kNN at layers 8 / 10 / 12) therefore runs via cloud-GPU rental (vast.ai / RunPod, ~$1, ~30 min wall-clock) using TransformerLens directly — no main-machine GPU work, no API dependency on the central bet. Goodfire's open-source SAE weights ([`Goodfire/Llama-3.1-8B-Instruct-SAE-l19`](https://huggingface.co/Goodfire/Llama-3.1-8B-Instruct-SAE-l19)) are downloadable and may underwrite a Method 6′ — *kNN over SAE-feature space at layer 19* — as a v0.2 sensitivity branch in the same rental session. Method 6′ is **not** a v0.1 substitute: it tests a different scientific question (feature-space kNN ≠ residual-stream kNN) and would need its own pre-registration before scoring.

Vision archived at tag [`v0.1-vision-archive`](../../releases/tag/v0.1-vision-archive). Where this came from: [`docs/philosophy.md`](./docs/philosophy.md).
