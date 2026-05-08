<div align="center">
  <img src="header.png" alt="Flywheel" width="256"/>
  <h1>Flywheel Geometry</h1>
  <p><strong>Bridge finder for personal knowledge bases.</strong></p>
  <p><em>What's structurally adjacent to what I'm thinking about — even if the language differs?</em></p>
</div>

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

Standard search is fine for "find me the note about X." It struggles at the question that actually compounds in a knowledge vault: *what's structurally adjacent to what I'm thinking about, even if the language differs?* — bridges across domains, where the same idea wears different vocabulary.

Recent interpretability work shows neural networks across architectures encode meaning on curved manifolds: helices, circumplexes, shells ([Lubana et al., 2025–2026](https://www.youtube.com/watch?v=F9eEYWX64ZA); [@slashreboot, 2026](https://doi.org/10.5281/ZENODO.18176077)). Cosine similarity cuts through the void between those shapes; geodesic distance follows the curve. The hypothesis: activation-derived geometry should surface non-obvious cross-domain bridges that strong embedding retrieval misses.

This repo tests that hypothesis empirically. **Normal search stays.** Geodesic mode is an additional axis answering the bridge-finding question — not a replacement for cosine retrieval. Whether the geodesic axis is worth shipping is the open empirical question. See **How we'll know this works** below.

---

## How this slots in

Flywheel Geometry is an **extension library for [flywheel-memory](https://github.com/velvetmonkey/flywheel-memory)** — the local-first MCP server that turns an Obsidian vault into safe AI memory. Today flywheel-memory ships hybrid BM25 + semantic search via Reciprocal Rank Fusion. Flywheel Geometry adds a *bridge-finder* axis: a separate query mode answering "what's structurally adjacent across domains?" alongside the existing precision-search modes. If the empirical experiment lands, the new axis ships as an optional layer — same MCP surface, additional `mode: "bridge"` discriminator, no replacement of existing search.

Part of the [Flywheel](https://github.com/velvetmonkey) suite:

- **[flywheel-memory](https://github.com/velvetmonkey/flywheel-memory)** — local-first MCP server, hybrid search, knowledge graph (this repo's host)
- **[flywheel-crank](https://github.com/velvetmonkey/flywheel-crank)** — Obsidian plugin: graph sidebar, vault health, semantic search UI
- **[flywheel-ideas](https://github.com/velvetmonkey/flywheel-ideas)** — falsifiable decision ledger with multi-model AI council dissent
- **flywheel-geometry** *(this repo)* — geodesic retrieval, in-progress

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

The currently tracked assumptions:

1. **Self-reported (x,y,z) coordinates carry retrieval-useful relational structure**, even though the [Zenodo reasoning traces](https://doi.org/10.5281/ZENODO.18176077) show that the mechanism is text generation from learned discourse priors (PAD valence/arousal, Russell circumplex, colour-wheel framings) rather than direct activation readout. The narrower question — whether those narrative-derived coordinates correlate enough with activation-derived geometry to be useful as a cheap proxy — is what the cheap-probe + Method 6 comparison resolves. *Falsifier:* (a) cheap-probe adversarial sweep — relational distances must survive prompt-frame perturbation (variants A, B, E rank-correlation > 0.5 vs core); (b) TransformerLens activation-derived distances on the same concepts must correlate with self-reported distances above chance.
2. **Activation-derived geometry contains retrieval-useful structure that strong embeddings miss.** *Falsifier:* ≥20% precision@5 lift over the LLM-rerank baselines (direct rerank, rationale-augmented rerank, and rationale-augmented rerank with a stronger reranker) on the 27 cross-domain bridge queries that constitute the primary metric. Control queries are reported separately, never mixed into the headline.
3. **Coordinate stability across runs is measurement-grade**, not stable narrative priors. *Falsifier:* adversarial replication with false anchors and synthetic concept domains.
4. **Manifold proximity outperforms bridge-tension** (high embedding distance × high relational similarity) on cross-domain bridges. *Falsifier:* head-to-head on the same query corpus.
5. **Human-rated bridge value is not explained by generic embedding similarity + LLM rationale generation.** *Falsifier:* baseline 2 above — if raters prefer rationale-augmented embeddings at equal or higher rates, the manifold effect is presentation, not retrieval.

If 5 refutes, the project pivots — and the public pivot post is the launch. Watching your own thesis fail in the open is the strongest brand outcome the bet can produce.

---

## Theoretical Foundation

- **Cross-architecture convergence** — Gemma, Llama, and GPT independently develop similar geometric structures (hue wheels, temporal helices, emotional circumplexes). *([Matthew (@slashreboot)](https://x.com/slashreboot), *Zero-Shot Geometric Probing Reveals Universal Cognitive Manifolds in LLMs*, Jan 2026 — [doi:10.5281/ZENODO.18176077](https://doi.org/10.5281/ZENODO.18176077))*

- **Cross-modality convergence on physical reality** — Edamadaka, Yang, Li, Gómez-Bombarelli (MIT), *Universally Converging Representations of Matter Across Scientific Foundation Models* — [arXiv:2512.03750](https://arxiv.org/abs/2512.03750) (Dec 2025). ~60 foundation models across string-, graph-, 3D-atomistic, protein, and LLM architectures converge on shared representations of matter without coordinated training. Better-performing models converge more strongly; weaker ones scatter to architecture-specific manifolds. The strongest available evidence that the geometry is a property of the territory, not the map.

- **Platonic Representation Hypothesis** — Huh, Cheung, Wang, Isola, *The Platonic Representation Hypothesis* — [arXiv:2405.07987](https://arxiv.org/abs/2405.07987) (ICML 2024). Different model families converge toward a shared statistical model of reality `Z` as scale, data volume, and task diversity grow; cross-modal alignment increases with capability. The unifying claim this project applies to retrieval.

- **Geometry = behavior** — Representation geometry is a direct reflection of data statistics and model beliefs. To control behavior you must respect the geometry. Geodesic paths stay on the manifold; linear paths enter the void. *([Ekdeep Singh Lubana (@EkdeepL)](https://x.com/EkdeepL) et al., [Goodfire AI (@GoodfireAI)](https://x.com/GoodfireAI), 2025–2026 — [talk](https://www.youtube.com/watch?v=F9eEYWX64ZA))*

- **Marr's three levels** — Behavior, algorithms, and representations are reflections of each other because the model learned the world's distribution. *(David Marr, *Vision*, 1982)*

- **The tool gap** — *"We will probably need tools which can capture these geometries in a general fashion — something like a SAE but which respects nonlinear geometry."* *(Ekdeep Singh Lubana, Goodfire AI, 2026)*

---

## Roadmap

**Phase 0 — Research & Validation**
- Replicate Matthew's zero-shot geometric probing on sample vault content
- Validate coordinate stability across runs and adversarial prompting
- Implement counterfactual pair generation + DAS causal filter
- Compare manifold proximity vs cosine similarity — identify divergence points

**Phase 1 — Proof of Concept**
- Build manifold coordinate store alongside standard embeddings
- Implement geodesic proximity query layer
- Validate (or refute) cross-domain bridge finding on real vault content

**Phase 2 — Manifold Index**
- Full vault indexing with manifold coordinates
- Visualise vault topology: dense clusters = deep expertise, sparse = gaps
- Confidence weighting: how strongly is each concept encoded?

**Phase 3 — Bridge Finder**
- "You're thinking about X. These notes are nearby on the manifold."
- Cross-domain surfacing as primary retrieval mode
- Hallucination-resistant retrieval via probe-based confidence signals

**Phase 4 — Ship inside flywheel-memory**
- Geodesic retrieval as an optional index layer alongside BM25 + semantic
- New `search(action: query, mode: "geodesic")` discriminator on the existing MCP surface
- Vault topology visualisation surfaced through [flywheel-crank](https://github.com/velvetmonkey/flywheel-crank)
- The SAE alternative Goodfire named but hasn't built

See [`benchmark/`](./benchmark/) for the empirical plan.

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

v0.1 benchmark scaffolded: 50-note synthetic corpus, 30 cross-domain queries (27 primary + 3 control), gold-firewall split, evaluator with pre-scoring guards, baseline methods in [`benchmark/methods/`](./benchmark/methods/). BM25 sanity floor is the first scored row on the [running leaderboard](./benchmark/results/RESULTS.md); LLM-rerank baselines are next, pinned to a named model so the numbers are reproducible. Cheap-probe Phase 0 (introspective coordinate validation under adversarial controls) in flight. Method 6 (geodesic, activation-derived retrieval) pre-registered for v0.2.

**30-day milestone (2026-06-07).** Three concrete success criteria, all required:
1. Kill-product floor scored on the 27 primary queries, with at least one of {BM25, LLM rerank, rationale-augmented rerank} producing a non-degenerate baseline number that the geodesic method must clear by ≥20% precision@5.
2. Cheap-probe Phase 0 resolved — pass (introspective coords carry retrieval-useful relational structure under adversarial framing → cheap baseline status earned) or pivot (project moves to TransformerLens activation extraction as the only path to genuine geometry).
3. ≥1 substantive reply from researcher outreach (Matthew, Ekdeep, or Thomas Fel) on the benchmark design.

If two of three slip, the public pivot post is the launch. Watching the thesis fail in the open is itself a defensible outcome.

Vision archived at tag [`v0.1-vision-archive`](../../releases/tag/v0.1-vision-archive). Where this came from: [`docs/philosophy.md`](./docs/philosophy.md).
