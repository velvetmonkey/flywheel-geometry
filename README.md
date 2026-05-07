# Flywheel Geometry

> *Your second brain, but you can see its shape.*

Most knowledge tools treat your notes as a flat list of words. Search finds keywords. Graphs show links. But meaning isn't flat — it has shape.

Neural networks trained on different data, different architectures, by different teams keep converging on the same internal geometry. That geometry isn't a property of the network. It propagates from the world through human perception, through language, through training data, into activations. It belongs to the bottom of that chain: reality itself.

Current embedding-based retrieval (cosine similarity on flat vectors) is leaving most of that signal on the table.

**Flywheel Geometry** maps your personal knowledge onto the manifold. It finds connections you couldn't have found yourself.

---

## The Problem

Standard retrieval measures cosine similarity between embedding vectors — linear distance in high-dimensional space. But concepts aren't encoded linearly. Neural networks encode meaning as curved manifolds: hue wheels, temporal spirals, helices, emotional circumplexes.

Steering linearly between two concepts cuts through the void where no valid meaning exists. The model becomes incoherent. Cosine similarity does the same to retrieval — it measures distance through regions that don't correspond to real semantic relationships.

Sparse autoencoders (SAEs) tile the manifold into fragments. They shatter the helix into disconnected shards. You get pieces of the shape, not the shape itself.

---

## The Solution

Flywheel Geometry captures **manifold coordinates** for each note — not just embedding vectors. Geometric position in the model's actual semantic space. Then queries by geodesic proximity: distance *along the curve*, not across the void.

**The killer feature:** cross-domain bridge finding. Notes from entirely different domains — horse training, AI architecture, finance, philosophy — that are geometrically adjacent surface automatically. No shared keywords. No explicit links. Just semantic proximity in the space where meaning actually lives.

---

## Theoretical Foundation

- **Cross-architecture convergence** — Gemma, Llama, GPT independently develop identical geometric structures (hue wheels, temporal helices, emotional circumplexes) without coordination. The geometry isn't in the network. It's in the world. *([Matthew (@slashreboot)](https://x.com/slashreboot), Zero-Shot Geometric Probing Reveals Universal Cognitive Manifolds in LLMs, Jan 2026 — [doi:10.5281/ZENODO.18176077](https://doi.org/10.5281/ZENODO.18176077))*

- **Geometry = behavior** — Representation geometry is a direct reflection of data statistics and model beliefs. To control behavior you must respect the geometry. Geodesic paths stay on the manifold; linear paths enter the void. *([Ekdeep Singh Lubana (@EkdeepL)](https://x.com/EkdeepL) et al., [Goodfire AI (@GoodfireAI)](https://x.com/GoodfireAI), 2025–2026 — [talk](https://www.youtube.com/watch?v=F9eEYWX64ZA))*

- **Marr's three levels** — Behavior, algorithms, and representations are reflections of each other because the model learned the world's distribution. The geometry propagates upward from reality through all three levels. *(David Marr, Vision, 1982)*

- **The tool gap** — *"We will probably need tools which can capture these geometries in a general fashion — something like a SAE but which respects nonlinear geometry."* *(Ekdeep Singh Lubana, Goodfire AI, 2026)*

---

## How It Works

**Probing method** (via [@slashreboot](https://x.com/slashreboot)):
A single message asks the model to introspect its own embedding space and return 3D coordinates (x,y,z) from a fixed central anchor. No activation extraction. No UMAP pipeline. The model self-reports its geometry. Coordinates are stable across runs and across models. Stored alongside standard embeddings.

**Causal filter** (via Geiger et al. / Goodfire):
Not all geometry is real. Counterfactual pairs are generated for each concept, then DAS (Distributed Alignment Search) identifies causally efficacious subspaces — those that actually drive meaning, not just correlate with it. Only those coordinates are stored.

**Retrieval**:
Query by geodesic proximity on the manifold, not cosine similarity on flat vectors. Surface notes from the neighbourhood of your current thinking, across all domains.

---

## Roadmap

**Phase 0 — Research & Validation**
- Replicate Matthew's zero-shot geometric probing on sample vault content
- Validate coordinate stability across runs
- Implement counterfactual pair generation + DAS causal filter
- Compare manifold proximity vs cosine similarity — identify divergence points

**Phase 1 — Proof of Concept**
- Build manifold coordinate store alongside standard embeddings
- Implement geodesic proximity query layer
- Validate cross-domain bridge finding on real vault content

**Phase 2 — Manifold Index**
- Full vault indexing with manifold coordinates
- Visualise vault topology: dense clusters = deep expertise, sparse = gaps
- Confidence weighting: how strongly is each concept encoded?

**Phase 3 — Bridge Finder**
- "You're thinking about X. These notes are nearby on the manifold."
- Cross-domain surfacing as primary retrieval mode
- Hallucination-resistant retrieval via probe-based confidence signals

**Phase 4 — Product**
- Manifold-aware search as a layer over any knowledge base
- Open source geometry-respecting manifold capture tool
- The SAE alternative Goodfire named but hasn't built

---

## Key References

- Ekdeep Singh Lubana et al. — *Belief Update as a Unifying Lens for In-Context Learning and Activation Steering* (Goodfire AI, 2025)
- Ekdeep Singh Lubana et al. — *Features as Rewards: Scalable Supervision for Open-Ended Tasks via Interpretability* (Goodfire AI, 2026)
- Ekdeep Singh Lubana et al. — nonlinear geometry + geodesic steering (Goodfire AI, upcoming 2026)
- Matthew (@slashreboot) — *Zero-Shot Geometric Probing Reveals Universal Cognitive Manifolds in Large Language Models* — [doi:10.5281/ZENODO.18176077](https://doi.org/10.5281/ZENODO.18176077) (Jan 2026)
- Yasaman Bahri et al. — theoretical prediction of geometry from data statistics
- Adakus Geiger et al. — causal abstraction / DAS framework (Goodfire AI)
- Goodfire AI — *[The World Inside Neural Networks](https://www.goodfire.ai/research/the-world-inside-neural-networks)* (2025)
- David Marr — *Vision* (1982)

---

## Status

Research stage. No implementation yet. Concept captured 7 May 2026 following Goodfire's neural geometry research drop and conversations with researchers in the space.

Interested in collaborating or following progress? Open an [issue](../../issues) or reach out.

---

*Part of the [Flywheel](https://github.com/velvetmonkey) ecosystem.*
