# Method 6 (v0.2): Activation extraction via TransformerLens + kNN — geodesic retrieval

**Status: pre-registered + scaffolded.** Pipeline split into two stages (`extract.py` for cloud-GPU forward passes, `run.py` for local kNN + retrieval). Backend stub awaits the rental session that fills in the TransformerLens hooks. This file pre-registers the design before any results land, so the bench can't be tuned to favour the hypothesis post-hoc.

## Pre-registration (locked 2026-05-08)

Per roundtable rounds 1–4 on the Method 6 plan:

- **Primary layer = layer 10** (median of the locked 8–12 band). Layers 8 and 12 reported as **sensitivity rows**, never substituted for layer 10 in the assumption resolution.
- **Negative control** = random-kNN row at layer 10. Same kNN topology with note IDs deterministically shuffled (seed 42) before edges are computed. Within-method floor: if layer 10 ≈ random-kNN, the geometry isn't doing work.
- **Hard gates for validation** (assumption `asm-3zmj1VGB` resolves "validate") — both required:
    1. `layer-10 p@5 ≥ 1.20 × Method 3 p@5` (currently 1.20 × 0.370 = 0.444)
    2. `layer-10 p@5 > random-kNN p@5 + 0.05 absolute`
- **Reportable, not gates**: layer-8 + layer-12 sensitivity rows; absolute lift over Method 3; graph-structural diagnostic (edge count, diameter, distance distribution); flag if diameter ≤ 2 or > 90 % of pairs within 2 hops (graph collapse — corpus finding, not Method 6 failure).
- **k = 10 fixed.** Frozen precision/model TBD by the rental spike (bf16 expected; int4 fallback if VRAM forces it).

## Activation artifact contract (locked 2026-05-08)

Stage-1 (`extract.py`, rental) and Stage-2 (`run.py`, this VM) communicate through one persistence format. Either side that diverges from this contract fails the manifest's reproducibility check.

| Field | Value |
|---|---|
| Model identity | `meta-llama/Llama-3.1-8B-Instruct` + HuggingFace revision SHA recorded in manifest |
| Layers | residual-stream `blocks.{8,10,12}.hook_resid_post` (TransformerLens naming) |
| Token selection / pooling | last-token of `f"{title}\n\n{body}"` for corpus notes; last-token of raw query string for queries |
| Tokenizer | model's default chat-template tokenizer; **no system prompt**; **no instruct-template wrapping** — both corpus notes and queries go in as raw user text in identical format |
| Normalisation | none — raw bf16 tensors. L2-normalisation considered an ablation, not the mainline |
| Persistence format | `np.savez_compressed(out, corpus_ids=..., query_ids=..., corpus_layer_8=..., corpus_layer_10=..., corpus_layer_12=..., query_layer_8=..., query_layer_10=..., query_layer_12=..., manifest=json)` |
| Dtype on disk | `float32` (bf16 not natively supported in `.npz`; cast happens at extraction) |
| Distance metric | Euclidean for kNN edge weights; shortest-path on resulting symmetric-union graph |

**Acceptance test**: Stage 2 reads the `.npz`, asserts `corpus_layer_10.shape == (50, 4096)` and `query_layer_10.shape == (30, 4096)`, builds a kNN graph, exits with no errors. If shape or dtype drifts, Stage 2 raises and refuses to score.

## The hypothesis being tested

For cross-domain bridge queries — where the *same idea* is stated in different vocabulary across domains — retrieval over a kNN graph in residual-stream activation space surfaces useful bridges that strong embedding retrieval (Voyage-3 + Claude rerank, with or without rationale generation) misses.

## Pre-registered design

### Model

**Phase 0 + v0.2 mainline (locked 2026-05-08): `meta-llama/Llama-3.1-8B-Instruct`.** Replication run on `google/gemma-2-9b` for cross-family robustness once the mainline holds. Pinned by:
- model name + size (e.g., `meta-llama/Llama-3.1-8B-Instruct`, `google/gemma-2-9b`)
- exact weight digest where exposed (HuggingFace revision SHA)
- precision (bf16 preferred; int8 acceptable on consumer hardware)

Reasoning for the smaller model: Phase 0 + Phase 1 plumbing must run on consumer hardware for reproducibility, the Llama-3 family has the most existing TransformerLens / interpretability tooling, and 8B-class models are well-represented in the activation-geometry literature (Goodfire helices, Lubana belief manifolds, the public probe-quality datasets). A larger model (Llama-3.3-70B / Gemma-3-27B) is added in v0.3+ only after the mainline has cleared the 27 cross-domain queries — robustness check, not a primary measurement.

### Layer

Mainline: a single residual-stream layer in the **8–12 band** (mid-network for a 32-layer 8B model — where the Llama-3 family surfaces abstract semantic/relational structure most cleanly per published probing work). Final layer chosen on a *held-out probe-quality dataset* (small handcrafted concept-pair adjacency set), **not** the 30-query bridge corpus.

Reserved variants:
- **Layer 16** — strong secondary if mainline is inconclusive on the held-out probe set
- **Layer 0** (early/embedding) — syntactic-control sanity check; geometry should *not* be present here, used as a negative control

The chosen layer is published in the v0.2 manifest before any retrieval scoring runs.

### Pooling

- **Mainline**: last-token pooling on a `f"{title}\n\n{body}"` formatted note. Matches the autoregressive forward pass conditioning.
- **Ablation**: mean pooling over all tokens.

Both reported.

### Distance

- **Mainline**: Euclidean distance in raw activation space → kNN graph (k=10 by default) → shortest-path graph distance for retrieval.
- **Ablation 1**: cosine distance in raw activation space → same kNN graph → shortest path.
- **Ablation 2**: diffusion distance on the kNN graph (random-walk based; smoother than shortest-path).
- **Ablation 3**: principal-curve fitting on identified manifold structures (helices, circles); geodesic distance along the curve. Only applied to subsets of the corpus where prior structural inspection has identified a known geometric form.

The mainline is shortest-path, not diffusion or principal-curve, because it's the simplest version of "graph-aware geodesic" and adds the least researcher-degree-of-freedom.

### kNN graph parameters

- **k**: 10 by default. Sweeps {5, 10, 20, 50} reported as ablations.
- **Edge weights**: Euclidean activation distance.
- **Symmetry**: kNN graph is symmetrised by union (edge from A to B if A ∈ kNN(B) *or* B ∈ kNN(A)).

### What does NOT appear in the mainline

- **No UMAP in the retrieval substrate**. UMAP is for visualisation only — it distorts global structure in ways that hurt retrieval. If a UMAP-based variant is added later, it's a sub-baseline, not the headline.
- **No introspective probe coordinates**. Method 5 uses those as a separate cheap baseline.
- **No DAS / causal-efficacy filter**. That comes in Phase 2 (subspace identification), gated on Phase 1 success.
- **No dimensionality reduction prior to kNN**. Activations are kept at full dimension; kNN computes in original space.

## Reproducibility

When `run.py` lands, the manifest must capture:
- HuggingFace model + revision SHA + parameter precision
- Selected layer index
- Pooling strategy
- Distance metric + graph algorithm
- kNN parameters
- Random seeds (for any stochastic steps — none expected in mainline)
- Wall-clock + GPU type
- Full corpus + queries SHA-256

Any deviation from this spec at run time gets logged as a **protocol violation** in the manifest, with explanation.

## Decision criterion

(Inherited from `benchmark/README.md`.) Method 6 ships if it shows ≥20% precision@5 lift over **both** Method 3 and Method 4a (or 4b, whichever performs better) on the **27 cross-domain bridge queries**. Equal or worse vs Methods 4a/4b → pivot.

## Open design questions (to resolve before run.py lands)

1. **Probe-quality dataset for model + layer choice**. Need a small held-out concept-probing corpus that's distinct from the bridge benchmark. Candidates: the 12-domain probe set from @slashreboot's Zenodo bundle (out of distribution from our notes); a handcrafted small probe set on the same domain types as the bridge corpus.
2. **Token pooling vs token-level retrieval**. Single-vector-per-note retrieval (default) vs token-level kNN with aggregation. Token-level is more expensive but may capture finer structure. Decide before run.
3. **GPU budget**. Llama-3.1-8B-Instruct in bf16 ≈ 16GB → fits a single 24GB consumer card (RTX 4090 / A5000) with room for the kNN graph and TransformerLens hooks. Replication on Gemma-2-9B fits the same hardware. Larger-model robustness check (Llama-3.3-70B / Gemma-3-27B) is v0.3+ and needs A100 80GB × 2 or H100 80GB — gated on v0.2 retrieval lift landing first.
