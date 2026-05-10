# Trial 2 Postmortem — Method 6 (residual-stream geodesic) FAIL

**Date run:** 2026-05-10
**Locked at:** commit `9bc52896fb` (`chore: standardize Flywheel Suite cross-link section`)
**Verdict:** the pre-registered falsifier ran. It failed. By a factor of ~4×.

---

## Bottom line

We pre-registered a benchmark in this repo on 2026-05-08 with a hard ship criterion: Method 6 (kNN + geodesic distance over residual-stream activations of `meta-llama/Llama-3.1-8B-Instruct` at layers 8/10/12) had to clear **≥ 1.20 × Method 3's primary p@5** on the 27 cross-domain bridge queries — i.e. ≥ 0.444 absolute precision@5 — *and* beat its own random-kNN control by ≥ 0.05 absolute.

The locked benchmark ran on a one-shot cloud-GPU rental (vast.ai RTX 4090, ~$0.17 spend). Layer-10 (primary) returned **P@5 = 0.104**. It lost to BM25 (0.252) by a factor of 2.4×, to voyage-native-rerank (0.333) by 3.2×, and to the kill-product floor `cli-direct-rerank-claude-sonnet` (0.370) by 3.6×. Within the same activation space, the geodesic graph beat its random-kNN control by only **+0.015 absolute** — statistically tied at n=27.

The pre-registered hypothesis is refuted on this benchmark with this model and these queries. We do not get to retest, retune, or relabel. That is what pre-registration costs.

---

## What was tested

| Element | Spec |
|---|---|
| Method | kNN graph over residual-stream activations + geodesic shortest-path retrieval |
| Model | `meta-llama/Llama-3.1-8B-Instruct`, bfloat16 |
| Layers | 8, 10 (primary), 12 (sensitivity) |
| Corpus | 50 synthetic notes spanning 5 domains (`benchmark/corpus/notes.jsonl`) |
| Queries | 30 cross-domain bridge queries (27 primary scored, 3 control) (`benchmark/queries.public.jsonl`) |
| Gold | Hidden ground-truth target sets per query (`benchmark/gold/targets.jsonl`) |
| Statistic | precision@5 on primary 27, vs locked baselines |
| Within-method control | Random-kNN at primary layer 10, seed 42 |
| Hard ship gates | layer-10 P@5 ≥ 1.20 × Method 3 primary p@5 (≥ 0.444 absolute) AND > random-kNN + 0.05 |

The benchmark scaffolding — corpus, queries, gold, eval, baseline scores — was committed to `main` on 2026-05-08 *before* the activation extraction code was even runnable on hardware. That ordering is the core of the pre-registration: when Method 6 ran, it was scored against bars that had been published for two days.

---

## Result

The leaderboard at `benchmark/results/RESULTS.md`, after Trial 2 was scored:

| Method | P@5 (primary, n=27) | nDCG@5 | Notes |
|---|---|---|---|
| `bm25-2026-05-08` | 0.252 | 0.439 | Sanity floor |
| `cli-direct-rerank-claude-sonnet-2026-05-08` | **0.370** | 0.670 | Kill-product floor |
| `voyage-native-rerank-2026-05-08` | 0.333 | 0.624 | Strong embedding baseline |
| `method-geodesic-llama31-8b-trial2-layer10-2026-05-10` (PRIMARY) | **0.104 (11/27)** | 0.154 | Method 6 |
| `method-geodesic-llama31-8b-trial2-layer10-randomctrl-2026-05-10` | 0.089 (12/27) | 0.127 | Within-method random-kNN control |
| `method-geodesic-llama31-8b-trial2-layer12-2026-05-10` | 0.074 | 0.105 | Sensitivity |
| `method-geodesic-llama31-8b-trial2-layer8-2026-05-10` | 0.059 | 0.089 | Sensitivity |

Three observations the data forces:

1. **Method 6 loses to BM25.** The cheapest possible baseline beat the hypothesis by 2.4×. Whatever signal residual-stream kNN was supposed to surface that lexical overlap couldn't, did not surface here.
2. **Layer-10 geodesic is statistically indistinguishable from random-kNN at the same layer.** +0.015 absolute is within the eval script's `tied_with_best` threshold (0.05 absolute) — graph structure adds essentially nothing over random walks in the same activation space. The eval flagged the tie.
3. **No clean layer trend.** Layer 12 (deeper) is *worse* than layer 10, layer 8 is the worst. The pre-registered intuition that deeper layers carry more abstract concept geometry did not survive contact with this corpus.

---

## What this refutes

- **The locked v0.1 ship criterion.** Method 6 cannot ship at v0.1. The benchmark, the gates, and the baselines all behaved as designed; the hypothesis lost cleanly.
- **The pre-registered open assumption `asm-3zmj1VGB`** ("activation-derived geometry contains retrieval-useful structure that strong embeddings miss") — refuted on this benchmark, with this model, with these queries, against these baselines.
- **Within the chosen activation space, graph topology over kNN edges adds no measurable signal over random walks.** The geodesic structure isn't carrying retrieval information that random-kNN doesn't already capture.

## What this does NOT refute

The discipline of pre-registration cuts both ways. The benchmark refutes the specific locked claim; it does not entitle us to broader killer narratives.

- It does not show that activation-derived geometry doesn't exist or isn't interesting in other respects.
- It does not test SAE-decoded feature space (the deferred Method 6′ branch — `Goodfire/Llama-3.1-8B-Instruct-SAE-l19`). Failing in the residual stream is silent on the SAE-feature-space hypothesis.
- It does not test other model families, other layer choices outside 8/10/12, or other query distributions.
- It does not test whether activation geometry helps on real personal vaults (this benchmark is synthetic; the corpus was authored for the test).
- It does not refute the broader telescope-fidelity / instrument-fidelity research programme, *but* combined with the earlier cheap-probe-360 result (2026-05-08, refuted on the introspective-coordinate side) it is the second clean fail in the same family of methods. Two strikes is signal.

---

## Provenance

| Field | Value |
|---|---|
| Locked SHA | `9bc52896fbaa25a5c50d27f629a30fdd1608c082` |
| Trial date | 2026-05-10 |
| GPU | NVIDIA GeForce RTX 4090 (24 GB), driver 535.154.05 |
| Datacenter | vast.ai instance 36474603, machine 15480, datacenter 1647 (Iceland) |
| torch | `2.4.0+cu121` |
| transformer_lens | `2.8.0` |
| transformers | `4.45.2` |
| accelerate | `0.34.2` |
| huggingface_hub | `0.25.1` |
| Wall-clock | model load 21.3 s · corpus extract 5.5 s · query extract 2.1 s · Stage 2 (kNN+geodesic) 0.6 s |
| Activations SHA256 | `bf619cf5151f0d19141b7f2b5d49b83b8768c06fa39d005ae2ab65438a087b7e` |
| Corpus SHA256 (locked) | `76f33934a2d994c3d77e0611b3ceb5209c8a972836fc164aa8291d761b2a0d7d` |
| Queries SHA256 (locked) | `e98ce569222310e06510ef676f994f73b5a66387c844e04315366cb2e075da07` |
| Gold SHA256 (locked) | `773b967a6784912990f8341fa596074cab17af6f9db6e6a811b0491bbc70b336` |
| Cost | ~$0.17 actual rental spend |

The corpus/queries/gold SHA256s captured in the local repo before the rental session match the SHA256s the rental's `extract.py` recorded into the activation manifest. Round-trip integrity: nothing on the rental swapped the inputs.

### One patch noted

Locked `extract.py` reads `transformer_lens.__version__` for the activation manifest. TL 2.8.0 dropped that attribute. A wrapper at `/tmp/run_extract.py` (now committed for audit alongside the activations) monkey-patched `transformer_lens.__version__` from `importlib.metadata.version("transformer_lens")` before invoking the unmodified locked `extract.py` via `runpy.run_path`. The manifest records `transformer_lens_version: "2.8.0"` — same string the locked code would have written if the attribute had still existed. Activations are byte-identical to what unpatched code would produce; only a manifest field-population path changed. Locked SHA `9bc52896fb` is untouched.

This is documented here, in `benchmark/methods/method-geodesic/cache/run_extract.py` (the wrapper), and in the corresponding vault note `[[trial2-verdict-2026-05-10]]`. It is not a pre-registration violation; it is the smallest possible workaround for a stale dependency attribute, with provenance preserved.

---

## Survivors and kill list

**Survives** (folds forward):

- The benchmark itself — corpus, queries, gold, baselines, eval. Locked, scored, public. Future researchers can run their own methods against the same hidden gold.
- The kill-product baseline framing. The eval did its job: it told us BM25 was already at 0.252, that voyage-native-rerank was 0.333, and that the LLM-rerank floor was 0.370. Without these, "Method 6 = 0.104" would be naked. With them, it's a verdict.
- The instrument-fidelity / telescope-fidelity research framing — but as a *question now flagged twice*. Cheap-probe-360 refuted it on the introspective-coordinate side (2026-05-08). Method 6 refuted it on the activation-extraction side (2026-05-10). The framing as *open question* survives; the framing as *load-bearing claim* does not.

**Kill list** (do not pursue, do not relitigate):

- Method 6 as v0.2 ship criterion. Refuted. Not promotable on this benchmark.
- "Manifold-aware retrieval beats embedding-and-rerank retrieval" as the primary product hypothesis for the broader Flywheel suite.
- Any framing that retroactively makes this benchmark not the falsifier — e.g. "the benchmark was the wrong test" or "the corpus was unfair" or "the model was too small". The benchmark was locked, the corpus and gold were authored before the run, the model was the one named in the pre-registration. Moving any of these post-hoc is a violation in spirit even with the bytes locked.
- Extending Trial 2 with more layers, more models, or more queries to "rescue" the result.

**Deferred** (not killed; awaiting separate decisions):

- Method 6′ — kNN over SAE-feature space at layer 19 of the same model, using `Goodfire/Llama-3.1-8B-Instruct-SAE-l19`. Different feature space, different test. Failing in residual-stream is silent on this. Auto-promoting Method 6′ would conflate two distinct hypotheses; auto-killing it would conflate them in the other direction. The honest stance is to keep it as a separately pre-registerable branch, not a substitute.
- The sibling project `flywheel-concept` — its bridge claim depends partially on the same activation-extraction substrate Method 6 just failed in. Whether to reshape, kill, or stay-the-course is its own decision, made on its own evidence base, not collapsed into this postmortem.

---

## Cascade — how this routes the rest of the work

The pre-registration encoded a FAIL branch explicitly. That branch is now active.

- **`flywheel-geometry` itself** — closed at v0.1 with the falsifier ran-and-failed outcome as the canonical artifact. Phases 2 (Manifold Index), 3 (Bridge Finder), and 4 (Product) are marked moot in the roadmap. Future work on this repo, if any, is bug-fix-only on the locked benchmark, plus this postmortem.
- **Witness Theory Paper 2** — was gated on Method 6 PASS. Retires as a paper; the negative-result writeup is what you are reading now.
- **Witness Theory Paper 1** — continues, but the AI-telescope / activation-manifold rhetoric in the draft must be stripped, not retained as "bridge premise only". Paper 1 is now defensible on neuroscience and philosophy alone.
- **Public posting (the "quiet-wait rule")** — lifted. Pre-drafted public material that was held until Trial 2 resolved is now publishable. The companion blog post `The Telescope Is Not the Stars` and a long-form X thread are the first two pieces in that queue.
- **`flywheel-concept`** — the sibling research programme — is deferred for its own decision (reshape-narrow / kill / stay-the-course). Whatever it lands on does not change the contents of this postmortem.

---

## What this earned

The strongest brand outcome the original 2026-05-07 council identified for this project was *"runs falsifiers on its own thesis."* That outcome required the falsifier to actually run. It did. The thesis lost cleanly. Total cost from pre-registration to verdict: under $1 of compute and ~25 minutes of one-shot rental wall-clock, with the discipline of the pre-registration intact.

Failing a thesis is cheap when the protocol is locked. Carrying a refuted thesis forward as if nothing happened is the expensive failure mode. The point of this repo was to make that mode unavailable. It worked.

---

## Reading list

- [`benchmark/results/RESULTS.md`](../benchmark/results/RESULTS.md) — full leaderboard with all per-query metrics
- [`benchmark/methods/method-geodesic/`](../benchmark/methods/method-geodesic/) — locked Method 6 implementation
- [`benchmark/results/method-geodesic-llama31-8b-trial2-*`](../benchmark/results/) — per-layer Trial 2 outputs, traces, graphml dumps, manifest, and within-method random-kNN control
- [`docs/v0.1-pivot.md`](./v0.1-pivot.md) — earlier pivot when the cheap-probe Phase 0 falsifier failed (2026-05-08)
- The flywheel-ideas decision ledger entry `idea-b4ZeRCoa` carries the assumption-and-outcome state (`asm-3zmj1VGB` now refuted, outcome assigned)
