"""Stage 1: residual-stream activation extraction via TransformerLens.

Runs on a rented GPU box. Loads Llama-3.1-8B-Instruct in bf16, forward-passes
the 50-note corpus + 30 queries, extracts last-token residual-stream tensors at
layers 8, 10, 12, saves to a single .npz.

Stage 2 (`run.py`, runs locally) reads the .npz, builds the kNN graph, computes
geodesic retrieval. The two stages communicate only through this file.

Activation artifact contract: see method.md.

Usage on rental:
    pip install torch transformers transformer_lens accelerate huggingface_hub
    export HF_TOKEN=hf_...
    python extract.py \\
        --corpus benchmark/corpus/notes.jsonl \\
        --queries benchmark/queries.public.jsonl \\
        --layers 8,10,12 \\
        --out activations-2026-05-08.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


def file_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", default="8,10,12",
                    help="Comma-separated residual-stream layer indices")
    ap.add_argument("--model", default=MODEL_NAME,
                    help="HuggingFace model id (locked spec uses Llama-3.1-8B-Instruct)")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"],
                    help="Model precision on GPU; output cast to float32 in .npz")
    args = ap.parse_args()

    layers = [int(s.strip()) for s in args.layers.split(",")]
    if not args.out.parent.exists():
        args.out.parent.mkdir(parents=True, exist_ok=True)

    # Heavy imports gated to avoid forcing torch on the local Stage-2 VM
    import numpy as np
    import torch
    from transformer_lens import HookedTransformer

    if not torch.cuda.is_available():
        print("ERROR: no CUDA device. Stage 1 requires a GPU.", file=sys.stderr)
        return 2

    if not os.environ.get("HF_TOKEN"):
        print("ERROR: HF_TOKEN env var unset. Required to download Llama-3.1-8B-Instruct.",
              file=sys.stderr)
        return 2

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    print(f"Loading {args.model} (dtype={args.dtype})...", flush=True)
    t_load_start = time.time()
    model = HookedTransformer.from_pretrained(
        args.model,
        dtype=dtype,
        device="cuda",
    )
    model.eval()
    t_load = time.time() - t_load_start
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    print(f"  loaded in {t_load:.1f}s; n_layers={n_layers}, d_model={d_model}", flush=True)

    for layer in layers:
        if not (0 <= layer < n_layers):
            print(f"ERROR: layer {layer} out of range (model has {n_layers} layers)",
                  file=sys.stderr)
            return 2

    hook_names = [f"blocks.{layer}.hook_resid_post" for layer in layers]

    def encode_one(text: str) -> dict[int, np.ndarray]:
        """Forward pass + last-token residuals at requested layers. Returns {layer: vec[d_model]}."""
        with torch.no_grad():
            _, cache = model.run_with_cache(text, names_filter=hook_names)
        return {
            layer: cache[f"blocks.{layer}.hook_resid_post"][0, -1, :]
                       .to(torch.float32).cpu().numpy()
            for layer in layers
        }

    corpus = [json.loads(line) for line in args.corpus.read_text().splitlines() if line.strip()]
    queries = [json.loads(line) for line in args.queries.read_text().splitlines() if line.strip()]
    print(f"  corpus: {len(corpus)} notes, queries: {len(queries)}", flush=True)

    corpus_ids = np.array([c["id"] for c in corpus])
    query_ids = np.array([q["query_id"] for q in queries])

    corpus_acts: dict[int, np.ndarray] = {layer: np.zeros((len(corpus), d_model), dtype=np.float32) for layer in layers}
    query_acts: dict[int, np.ndarray] = {layer: np.zeros((len(queries), d_model), dtype=np.float32) for layer in layers}

    print("Extracting corpus activations...", flush=True)
    t_corpus_start = time.time()
    for i, note in enumerate(corpus):
        text = f"{note['title']}\n\n{note['body']}"
        acts = encode_one(text)
        for layer in layers:
            corpus_acts[layer][i] = acts[layer]
        if (i + 1) % 10 == 0:
            print(f"  corpus {i + 1}/{len(corpus)} ({time.time() - t_corpus_start:.1f}s)", flush=True)
    t_corpus = time.time() - t_corpus_start
    print(f"  corpus done in {t_corpus:.1f}s", flush=True)

    print("Extracting query activations...", flush=True)
    t_query_start = time.time()
    for i, q in enumerate(queries):
        acts = encode_one(q["query_text"])
        for layer in layers:
            query_acts[layer][i] = acts[layer]
        if (i + 1) % 10 == 0:
            print(f"  query {i + 1}/{len(queries)} ({time.time() - t_query_start:.1f}s)", flush=True)
    t_query = time.time() - t_query_start
    print(f"  queries done in {t_query:.1f}s", flush=True)

    manifest = {
        "model": args.model,
        "dtype_runtime": args.dtype,
        "dtype_on_disk": "float32",
        "layers": layers,
        "n_layers_in_model": n_layers,
        "d_model": d_model,
        "pooling": "last-token",
        "corpus_format": 'f"{title}\\n\\n{body}"',
        "query_format": "raw query_text, no template",
        "corpus_count": len(corpus),
        "queries_count": len(queries),
        "corpus_sha256": file_checksum(args.corpus),
        "queries_sha256": file_checksum(args.queries),
        "wall_clock_seconds": {
            "model_load": round(t_load, 1),
            "corpus_extract": round(t_corpus, 1),
            "query_extract": round(t_query, 1),
        },
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "transformer_lens_version": __import__("transformer_lens").__version__,
        "torch_version": torch.__version__,
    }

    save_kwargs: dict = {
        "corpus_ids": corpus_ids,
        "query_ids": query_ids,
        "manifest": np.array(json.dumps(manifest)),
    }
    for layer in layers:
        save_kwargs[f"corpus_layer_{layer}"] = corpus_acts[layer]
        save_kwargs[f"query_layer_{layer}"] = query_acts[layer]

    np.savez_compressed(args.out, **save_kwargs)
    size_mb = args.out.stat().st_size / 1024 / 1024
    print(f"\nWrote {args.out} ({size_mb:.1f} MB)", flush=True)
    print(f"Manifest: {json.dumps(manifest, indent=2)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
