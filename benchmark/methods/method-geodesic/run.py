"""Stage 2: kNN graph + geodesic retrieval from pre-extracted activations.

Runs locally (this VM, no GPU). Reads the .npz produced by `extract.py` on a
rented GPU, builds a kNN graph per layer in raw activation space, computes
shortest-path distances, retrieves top-5 corpus notes per query.

Outputs one JSONL per layer (eval-format) plus a random-kNN negative control
row at the pre-registered primary layer. Each result row populates the
`manifold_uri` field. Per-layer kNN graphs are also dumped as .graphml for
optional visualisation.

See method.md for the activation artifact contract this script consumes.

Usage:
    python run.py \\
        --activations cache/activations-2026-05-08.npz \\
        --corpus ../../corpus/notes.jsonl \\
        --queries ../../queries.public.jsonl \\
        --out-prefix ../../results/method-geodesic-llama31-8b \\
        --k 10 \\
        --primary-layer 10 \\
        --random-control \\
        --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import networkx as nx
import scipy.sparse
import scipy.sparse.csgraph

METHOD_NAME = "method-geodesic"
PRIMARY_LAYER_DEFAULT = 10
RANDOM_SEED_DEFAULT = 42


def file_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_knn_graph(corpus_acts: np.ndarray, k: int) -> tuple[scipy.sparse.csr_matrix, np.ndarray]:
    """Build a symmetric-union kNN graph in raw activation space (Euclidean).

    Returns (sparse adjacency with edge weights = Euclidean distance, full N×N distance matrix).
    Tie-break is deterministic by note index (stable sort).
    """
    n = corpus_acts.shape[0]
    diff = corpus_acts[:, None, :] - corpus_acts[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))  # (N, N) Euclidean

    rows: list[int] = []
    cols: list[int] = []
    weights: list[float] = []
    for i in range(n):
        order = np.argsort(dist[i], kind="stable")  # ascending; self at 0
        chosen = [j for j in order if j != i][:k]
        for j in chosen:
            rows.append(i)
            cols.append(j)
            weights.append(float(dist[i, j]))

    # Symmetrise by union (edge present if A in kNN(B) OR B in kNN(A))
    edges = {}
    for r, c, w in zip(rows, cols, weights):
        a, b = (r, c) if r < c else (c, r)
        if (a, b) not in edges:
            edges[(a, b)] = w
        else:
            edges[(a, b)] = min(edges[(a, b)], w)  # tie-break to smaller distance

    sym_rows: list[int] = []
    sym_cols: list[int] = []
    sym_weights: list[float] = []
    for (a, b), w in edges.items():
        sym_rows.extend([a, b])
        sym_cols.extend([b, a])
        sym_weights.extend([w, w])

    adj = scipy.sparse.csr_matrix(
        (sym_weights, (sym_rows, sym_cols)), shape=(n, n), dtype=np.float64
    )
    return adj, dist


def graph_diagnostic(adj: scipy.sparse.csr_matrix) -> dict:
    """Edge count, avg degree, connected components, diameter, distance distribution."""
    n = adj.shape[0]
    G = nx.from_scipy_sparse_array(adj)
    n_edges = G.number_of_edges()
    degrees = [d for _, d in G.degree()]
    components = list(nx.connected_components(G))
    largest = max(components, key=len)
    sub = G.subgraph(largest)
    try:
        diameter = nx.diameter(sub) if len(largest) > 1 else 0
    except Exception:
        diameter = -1

    # Distribution of shortest-path hop counts (largest component only)
    shortest, _ = scipy.sparse.csgraph.shortest_path(
        adj, return_predecessors=True, unweighted=True
    )
    finite_pairs = shortest[(shortest != np.inf) & (shortest > 0)]
    hop_pcts = {}
    if finite_pairs.size:
        for h in (1, 2, 3, 4, 5):
            hop_pcts[f"pct_pairs_hops_{h}_or_less"] = float(
                (finite_pairs <= h).sum() / finite_pairs.size
            )

    return {
        "n_nodes": n,
        "n_edges": n_edges,
        "mean_degree": round(float(np.mean(degrees)), 2),
        "max_degree": int(np.max(degrees)) if degrees else 0,
        "min_degree": int(np.min(degrees)) if degrees else 0,
        "n_components": len(components),
        "largest_component_size": len(largest),
        "diameter_largest_component": diameter,
        "diameter_le_2_flag": bool(diameter <= 2 and diameter >= 0),
        "hop_distribution": hop_pcts,
    }


def retrieve_top_k(
    query_act: np.ndarray,
    corpus_acts: np.ndarray,
    geodesic: np.ndarray,
    top_k: int,
) -> tuple[list[int], int, float]:
    """Nearest corpus note in Euclidean → take its row of the geodesic matrix → top-k.

    Returns (top_k_indices, anchor_index, anchor_distance).
    """
    diffs = corpus_acts - query_act[None, :]
    eucl = np.sqrt((diffs ** 2).sum(-1))
    anchor = int(np.argmin(eucl))
    geodesic_row = geodesic[anchor]
    # Sort ascending by geodesic distance (stable; ties broken by corpus index)
    order = np.argsort(geodesic_row, kind="stable")
    top_indices = [int(j) for j in order[:top_k]]
    return top_indices, anchor, float(eucl[anchor])


def shuffle_corpus_for_random_control(corpus_acts: np.ndarray, seed: int) -> np.ndarray:
    """Permute the row order of corpus activations deterministically.
    Same kNN topology rebuilt from the shuffled rows produces the random-kNN control."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(corpus_acts.shape[0])
    return corpus_acts[perm]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", type=Path, required=True,
                    help=".npz produced by extract.py (Stage 1)")
    ap.add_argument("--corpus", type=Path, required=True,
                    help="benchmark/corpus/notes.jsonl — used for title lookup in JSONL output")
    ap.add_argument("--queries", type=Path, required=True,
                    help="benchmark/queries.public.jsonl")
    ap.add_argument("--out-prefix", type=Path, required=True,
                    help="results/method-geodesic-<model-tag>; per-layer suffix appended")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--k", type=int, default=10, help="kNN graph k")
    ap.add_argument("--primary-layer", type=int, default=PRIMARY_LAYER_DEFAULT)
    ap.add_argument("--random-control", action="store_true",
                    help="Also produce a random-kNN row at the primary layer")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED_DEFAULT)
    args = ap.parse_args()

    # Load .npz
    if not args.activations.exists():
        print(f"ERROR: activations file not found: {args.activations}", file=sys.stderr)
        return 2
    npz = np.load(args.activations, allow_pickle=False)
    corpus_ids = npz["corpus_ids"]
    query_ids = npz["query_ids"]
    extract_manifest = json.loads(str(npz["manifest"]))
    layers_in_npz = extract_manifest["layers"]
    print(f"Loaded {args.activations}: {len(corpus_ids)} corpus, {len(query_ids)} queries, "
          f"layers {layers_in_npz}", flush=True)

    # Sanity-check shapes against the activation contract
    for layer in layers_in_npz:
        c_key, q_key = f"corpus_layer_{layer}", f"query_layer_{layer}"
        c, q = npz[c_key], npz[q_key]
        d = extract_manifest["d_model"]
        assert c.shape == (len(corpus_ids), d), f"{c_key} shape {c.shape} != ({len(corpus_ids)}, {d})"
        assert q.shape == (len(query_ids), d), f"{q_key} shape {q.shape} != ({len(query_ids)}, {d})"
    print("  contract check: shapes OK", flush=True)

    # Load corpus + queries text for JSONL output
    corpus_notes = [json.loads(line) for line in args.corpus.read_text().splitlines() if line.strip()]
    queries = [json.loads(line) for line in args.queries.read_text().splitlines() if line.strip()]
    note_id_to_title = {n["id"]: n["title"] for n in corpus_notes}
    note_id_to_idx = {nid: i for i, nid in enumerate(corpus_ids)}
    query_id_to_text = {q["query_id"]: q["query_text"] for q in queries}
    note_ids_set = set(corpus_ids.tolist())

    # Output prefix path
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    started = time.time()
    artifacts: list[dict] = []

    for layer in layers_in_npz:
        print(f"\n=== Layer {layer} ===", flush=True)
        corpus_acts = npz[f"corpus_layer_{layer}"]
        query_acts = npz[f"query_layer_{layer}"]

        t0 = time.time()
        adj, _ = build_knn_graph(corpus_acts, args.k)
        diag = graph_diagnostic(adj)
        t_graph = time.time() - t0

        # Geodesic = shortest-path on the weighted symmetric kNN graph
        geodesic, _ = scipy.sparse.csgraph.shortest_path(
            adj, return_predecessors=True, directed=False
        )
        # Disconnected pairs come back as np.inf; replace with the max finite + 1 so they sort last
        finite_max = geodesic[np.isfinite(geodesic)].max() if np.isfinite(geodesic).any() else 0
        geodesic[~np.isfinite(geodesic)] = finite_max + 1

        # Retrieve per query
        rows: list[dict] = []
        traces: list[dict] = []
        for i, q in enumerate(queries):
            qid = q["query_id"]
            qtext = q["query_text"]
            top_indices, anchor, anchor_dist = retrieve_top_k(
                query_acts[i], corpus_acts, geodesic, args.top_k
            )
            top_ids = [str(corpus_ids[j]) for j in top_indices]
            top_k_records = [
                {"id": nid, "title": note_id_to_title.get(nid, "")}
                for nid in top_ids
            ]
            row = {
                "query_id": qid,
                "query_text": qtext,
                "top_k": top_k_records,
                "method": METHOD_NAME,
                "manifold_uri": (
                    f"manifold://{extract_manifest['model'].split('/')[-1].lower()}"
                    f"/layer-{layer}/run-{date_str}"
                ),
            }
            rows.append(row)
            traces.append({
                **row,
                "anchor_corpus_index": anchor,
                "anchor_corpus_id": str(corpus_ids[anchor]),
                "anchor_euclidean_distance": round(anchor_dist, 4),
                "geodesic_top_k_distances": [
                    round(float(geodesic[anchor, j]), 4) for j in top_indices
                ],
            })

        # Write JSONL
        out_jsonl = args.out_prefix.parent / f"{args.out_prefix.name}-layer{layer}-{date_str}.jsonl"
        with out_jsonl.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        # Write traces dir
        traces_dir = args.out_prefix.parent / "traces" / out_jsonl.stem
        traces_dir.mkdir(parents=True, exist_ok=True)
        for tr in traces:
            (traces_dir / f"{tr['query_id']}.json").write_text(json.dumps(tr, indent=2))

        # Diagnostic JSON
        diag_path = args.out_prefix.parent / f"{args.out_prefix.name}-layer{layer}-{date_str}-diagnostic.json"
        diag_path.write_text(json.dumps({
            "layer": layer,
            "k": args.k,
            "wall_clock_seconds_graph_build": round(t_graph, 2),
            **diag,
        }, indent=2))

        # GraphML dump
        graphml_path = args.out_prefix.parent / f"{args.out_prefix.name}-layer{layer}-{date_str}.graphml"
        G = nx.from_scipy_sparse_array(adj, edge_attribute="distance")
        nx.set_node_attributes(G, {i: str(corpus_ids[i]) for i in range(len(corpus_ids))}, name="note_id")
        nx.write_graphml(G, str(graphml_path))

        artifacts.append({
            "layer": layer,
            "jsonl": str(out_jsonl),
            "traces_dir": str(traces_dir),
            "diagnostic": str(diag_path),
            "graphml": str(graphml_path),
            "diagnostic_summary": {
                "n_edges": diag["n_edges"],
                "diameter": diag["diameter_largest_component"],
                "diameter_le_2": diag["diameter_le_2_flag"],
            },
        })
        print(f"  wrote {out_jsonl}, {graphml_path.name}, diagnostic", flush=True)
        print(f"  graph: {diag['n_edges']} edges, diameter {diag['diameter_largest_component']}", flush=True)

    # Random-kNN negative control at primary layer
    random_control_artifact = None
    if args.random_control:
        layer = args.primary_layer
        if layer not in layers_in_npz:
            print(f"WARN: primary layer {layer} not in extracted layers {layers_in_npz}; "
                  f"skipping random control", flush=True)
        else:
            print(f"\n=== Random-kNN control (layer {layer}, seed {args.seed}) ===", flush=True)
            corpus_acts = npz[f"corpus_layer_{layer}"]
            query_acts = npz[f"query_layer_{layer}"]
            shuffled = shuffle_corpus_for_random_control(corpus_acts, args.seed)

            adj, _ = build_knn_graph(shuffled, args.k)
            diag = graph_diagnostic(adj)
            geodesic, _ = scipy.sparse.csgraph.shortest_path(
                adj, return_predecessors=True, directed=False
            )
            finite_max = geodesic[np.isfinite(geodesic)].max() if np.isfinite(geodesic).any() else 0
            geodesic[~np.isfinite(geodesic)] = finite_max + 1

            rows = []
            traces = []
            for i, q in enumerate(queries):
                top_indices, anchor, anchor_dist = retrieve_top_k(
                    query_acts[i], shuffled, geodesic, args.top_k
                )
                top_ids = [str(corpus_ids[j]) for j in top_indices]
                top_k_records = [
                    {"id": nid, "title": note_id_to_title.get(nid, "")}
                    for nid in top_ids
                ]
                row = {
                    "query_id": q["query_id"],
                    "query_text": q["query_text"],
                    "top_k": top_k_records,
                    "method": f"{METHOD_NAME}-randomctrl",
                    "manifold_uri": (
                        f"manifold://{extract_manifest['model'].split('/')[-1].lower()}"
                        f"/layer-{layer}-randomctrl/run-{date_str}"
                    ),
                }
                rows.append(row)
                traces.append({**row, "anchor_corpus_index": anchor,
                               "anchor_euclidean_distance": round(anchor_dist, 4)})

            out_jsonl = args.out_prefix.parent / f"{args.out_prefix.name}-layer{layer}-randomctrl-{date_str}.jsonl"
            with out_jsonl.open("w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            traces_dir = args.out_prefix.parent / "traces" / out_jsonl.stem
            traces_dir.mkdir(parents=True, exist_ok=True)
            for tr in traces:
                (traces_dir / f"{tr['query_id']}.json").write_text(json.dumps(tr, indent=2))

            diag_path = args.out_prefix.parent / f"{args.out_prefix.name}-layer{layer}-randomctrl-{date_str}-diagnostic.json"
            diag_path.write_text(json.dumps({
                "layer": layer, "k": args.k, "control": "random-kNN",
                "seed": args.seed, **diag,
            }, indent=2))

            random_control_artifact = {
                "layer": layer,
                "jsonl": str(out_jsonl),
                "traces_dir": str(traces_dir),
                "diagnostic": str(diag_path),
            }
            print(f"  wrote {out_jsonl}", flush=True)

    # Top-level manifest
    manifest = {
        "method": METHOD_NAME,
        "started_at": datetime.fromtimestamp(started, tz=timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "wall_clock_seconds_total": round(time.time() - started, 2),
        "stage_1_extract_manifest": extract_manifest,
        "stage_2_run_args": {
            "activations": str(args.activations),
            "corpus_jsonl_sha256": file_checksum(args.corpus),
            "queries_jsonl_sha256": file_checksum(args.queries),
            "k": args.k,
            "top_k": args.top_k,
            "primary_layer": args.primary_layer,
            "random_control": args.random_control,
            "random_seed": args.seed,
        },
        "pre_registration": {
            "primary_layer": args.primary_layer,
            "sensitivity_layers": [l for l in layers_in_npz if l != args.primary_layer],
            "hard_gates": {
                "lift_over_method_3": "≥1.20×",
                "above_random_kNN": ">+0.05 absolute",
            },
        },
        "artifacts": artifacts,
        "random_control": random_control_artifact,
        "env": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "networkx": nx.__version__,
            "python": sys.version.split()[0],
        },
    }
    manifest_path = args.out_prefix.parent / f"{args.out_prefix.name}-{date_str}-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote manifest: {manifest_path}", flush=True)
    print(f"Total wall-clock: {manifest['wall_clock_seconds_total']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
