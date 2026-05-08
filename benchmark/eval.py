"""geodesic-bench evaluator.

Reads a method's results JSONL + the gold targets + the public queries, scores
precision@5 and nDCG@5 split by primary (27 cross-domain) and secondary (3 control)
queries, writes a per-method sidecar JSON, prints a markdown comparison table,
and appends one row per method to results/RESULTS.md.

Usage:
    python eval.py --gold gold/targets.jsonl --queries queries.public.jsonl \\
        results/<method1>-<date>.jsonl [results/<method2>-<date>.jsonl ...]

Pre-scoring guards (any failure → exit non-zero, no metrics emitted):
  1. Each results file has exactly 30 rows.
  2. No missing or duplicated query_ids within a file.
  3. No duplicate note IDs within any top_k array.
  4. Every retrieved ID exists in corpus/notes.jsonl.
  5. No method output contains target_note_ids/target_domains/rationale (gold-leak tripwire).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import date as _date
from pathlib import Path
from typing import Iterable

# v0.1-only debug aid; the real ship criterion remains ≥20% lift.
TIE_THRESHOLD_PRIMARY_PRECISION = 0.05

GOLD_LEAK_FIELDS = {"target_note_ids", "target_domains", "rationale"}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def precision_at_k(retrieved: list[str], gold: set[str]) -> float:
    if not retrieved:
        return 0.0
    return sum(1 for r in retrieved if r in gold) / len(retrieved)


def dcg_at_k(retrieved: list[str], gold: set[str]) -> float:
    return sum((1.0 if r in gold else 0.0) / math.log2(i + 2) for i, r in enumerate(retrieved))


def ideal_dcg_at_k(num_relevant: int, k: int) -> float:
    return sum(1.0 / math.log2(i + 2) for i in range(min(num_relevant, k)))


def first_relevant_rank(retrieved: list[str], gold: set[str]) -> int | None:
    for i, r in enumerate(retrieved):
        if r in gold:
            return i + 1  # 1-indexed
    return None


def guard_check(
    results: list[dict],
    method_path: Path,
    queries_by_id: dict[str, dict],
    corpus_ids: set[str],
) -> list[str]:
    """Return list of failure messages; empty list = all guards pass."""
    failures: list[str] = []

    # Guard 1: row count
    expected = len(queries_by_id)
    if len(results) != expected:
        failures.append(f"row count: expected {expected}, got {len(results)}")

    # Guard 2: query_id presence + uniqueness
    seen_ids: set[str] = set()
    for r in results:
        qid = r.get("query_id")
        if not qid:
            failures.append(f"row missing query_id: {r}")
            continue
        if qid in seen_ids:
            failures.append(f"duplicate query_id within file: {qid}")
        seen_ids.add(qid)
        if qid not in queries_by_id:
            failures.append(f"unknown query_id (not in queries.public.jsonl): {qid}")

    # Guards 3 + 4: top_k checks
    for r in results:
        qid = r.get("query_id", "<unknown>")
        top_k = r.get("top_k", [])
        # Permit either {id: ...} dicts or bare strings
        retrieved_ids = [item["id"] if isinstance(item, dict) else str(item) for item in top_k]
        if len(set(retrieved_ids)) != len(retrieved_ids):
            failures.append(f"{qid}: duplicate note IDs within top_k {retrieved_ids}")
        for rid in retrieved_ids:
            if rid not in corpus_ids:
                failures.append(f"{qid}: retrieved ID {rid!r} not in corpus")

    # Guard 5: gold-leak tripwire — no method output should contain gold fields
    raw_text = method_path.read_text()
    for field in GOLD_LEAK_FIELDS:
        if re.search(rf'"{re.escape(field)}"\s*:', raw_text):
            failures.append(
                f"gold-leak tripwire: results file contains field '{field}' — "
                "method runners must not read benchmark/gold/."
            )

    return failures


def score_method(
    results: list[dict],
    gold_by_id: dict[str, dict],
    queries_by_id: dict[str, dict],
    k: int = 5,
) -> dict:
    """Returns {per_query: [...], primary: {...}, secondary: {...}}."""
    per_query = []
    for r in results:
        qid = r["query_id"]
        gold_targets: list[str] = gold_by_id[qid]["target_note_ids"]
        gold = set(gold_targets)
        is_control = queries_by_id[qid].get("is_control", False)

        top_k = r.get("top_k", [])
        retrieved_ids = [item["id"] if isinstance(item, dict) else str(item) for item in top_k][:k]

        p_at_k = precision_at_k(retrieved_ids, gold)
        dcg = dcg_at_k(retrieved_ids, gold)
        idcg = ideal_dcg_at_k(len(gold_targets), k)
        ndcg = dcg / idcg if idcg > 0 else 0.0
        hits = [r_ in gold for r_ in retrieved_ids]

        per_query.append({
            "query_id": qid,
            "is_control": is_control,
            "gold_ids": gold_targets,
            "retrieved_ids_at_5": retrieved_ids,
            "hits_at_5": hits,
            "precision_at_5": p_at_k,
            "dcg_at_5": dcg,
            "ideal_dcg_at_5": idcg,
            "ndcg_at_5": ndcg,
            "first_relevant_rank": first_relevant_rank(retrieved_ids, gold),
            "manifold_uri": None,  # reserved for Method 6
        })

    primary = [q for q in per_query if not q["is_control"]]
    secondary = [q for q in per_query if q["is_control"]]

    def aggregate(rows: list[dict]) -> dict:
        if not rows:
            return {"n_queries": 0, "mean_precision_at_5": 0.0, "mean_ndcg_at_5": 0.0}
        return {
            "n_queries": len(rows),
            "mean_precision_at_5": sum(q["precision_at_5"] for q in rows) / len(rows),
            "mean_ndcg_at_5": sum(q["ndcg_at_5"] for q in rows) / len(rows),
            "queries_with_at_least_one_hit": sum(1 for q in rows if any(q["hits_at_5"])),
        }

    return {
        "per_query": per_query,
        "primary": aggregate(primary),
        "secondary": aggregate(secondary),
    }


def render_markdown_table(by_method: dict[str, dict], best_primary: float) -> str:
    headers = ["Method", "P@5 (primary 27)", "nDCG@5 (primary)", "P@5 (control 3)", "nDCG@5 (control)", "Tied w/ best?"]
    sep = " | "
    lines = [sep.join(headers), sep.join(["---"] * len(headers))]
    for name, eval_ in by_method.items():
        p_pri = eval_["primary"]["mean_precision_at_5"]
        n_pri = eval_["primary"]["mean_ndcg_at_5"]
        p_ctl = eval_["secondary"]["mean_precision_at_5"]
        n_ctl = eval_["secondary"]["mean_ndcg_at_5"]
        n_pri_q = eval_["primary"]["n_queries"]
        n_ctl_q = eval_["secondary"]["n_queries"]
        hits_pri = eval_["primary"].get("queries_with_at_least_one_hit", 0)
        hits_ctl = eval_["secondary"].get("queries_with_at_least_one_hit", 0)
        tied = "yes" if abs(p_pri - best_primary) <= TIE_THRESHOLD_PRIMARY_PRECISION else "no"
        lines.append(sep.join([
            name,
            f"{p_pri:.3f} ({hits_pri}/{n_pri_q})",
            f"{n_pri:.3f}",
            f"{p_ctl:.3f} ({hits_ctl}/{n_ctl_q})",
            f"{n_ctl:.3f}",
            tied,
        ]))
    return "\n".join(lines)


def render_results_md_row(name: str, eval_: dict) -> str:
    p_pri = eval_["primary"]["mean_precision_at_5"]
    n_pri = eval_["primary"]["mean_ndcg_at_5"]
    p_ctl = eval_["secondary"]["mean_precision_at_5"]
    n_ctl = eval_["secondary"]["mean_ndcg_at_5"]
    n_pri_q = eval_["primary"]["n_queries"]
    n_ctl_q = eval_["secondary"]["n_queries"]
    hits_pri = eval_["primary"].get("queries_with_at_least_one_hit", 0)
    hits_ctl = eval_["secondary"].get("queries_with_at_least_one_hit", 0)
    return (
        f"| `{name}` | "
        f"{p_pri:.3f} ({hits_pri}/{n_pri_q}) | "
        f"{n_pri:.3f} | "
        f"{p_ctl:.3f} ({hits_ctl}/{n_ctl_q}) | "
        f"{n_ctl:.3f} | "
        f"{_date.today().isoformat()} |"
    )


def append_or_replace_results_md(results_md: Path, method_name: str, row: str) -> None:
    if not results_md.exists():
        # Caller is expected to have scaffolded RESULTS.md. Refuse to create silently.
        raise FileNotFoundError(f"{results_md} does not exist; scaffold it first.")
    text = results_md.read_text()
    method_marker = f"| `{method_name}` |"
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    replaced = False
    in_leaderboard = False
    for line in lines:
        if line.startswith("<!-- leaderboard:start -->"):
            in_leaderboard = True
            out.append(line)
            continue
        if line.startswith("<!-- leaderboard:end -->"):
            if not replaced:
                out.append(row + "\n")
            in_leaderboard = False
            out.append(line)
            continue
        if in_leaderboard and line.startswith(method_marker):
            out.append(row + "\n")
            replaced = True
            continue
        out.append(line)
    results_md.write_text("".join(out))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="geodesic-bench evaluator")
    ap.add_argument("--gold", type=Path, required=True, help="Path to gold/targets.jsonl")
    ap.add_argument("--queries", type=Path, required=True, help="Path to queries.public.jsonl")
    ap.add_argument("--corpus", type=Path, default=Path(__file__).parent / "corpus" / "notes.jsonl")
    ap.add_argument("--results-md", type=Path, default=Path(__file__).parent / "results" / "RESULTS.md")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("results_files", nargs="+", type=Path,
                    help="One or more results JSONL files to score")
    args = ap.parse_args(argv)

    gold_by_id = {r["query_id"]: r for r in load_jsonl(args.gold)}
    queries_by_id = {r["query_id"]: r for r in load_jsonl(args.queries)}
    corpus_ids = {n["id"] for n in load_jsonl(args.corpus)}

    by_method: dict[str, dict] = {}
    any_failure = False
    for results_file in args.results_files:
        method_name = results_file.stem  # e.g. "bm25-2026-05-08"
        results = load_jsonl(results_file)

        failures = guard_check(results, results_file, queries_by_id, corpus_ids)
        if failures:
            any_failure = True
            print(f"\n[GUARD FAILURE] {method_name}", file=sys.stderr)
            for f in failures:
                print(f"  - {f}", file=sys.stderr)
            continue

        scored = score_method(results, gold_by_id, queries_by_id, k=args.top_k)
        by_method[method_name] = scored

        sidecar = results_file.with_name(results_file.stem + "-eval.json")
        sidecar.write_text(json.dumps({
            "method": method_name,
            "results_file": str(results_file.name),
            "primary": scored["primary"],
            "secondary": scored["secondary"],
            "tie_threshold_v01_only": TIE_THRESHOLD_PRIMARY_PRECISION,
            "per_query": scored["per_query"],
        }, indent=2))

    if any_failure:
        print("\n[FAIL] one or more guard checks failed; no metrics emitted for failing methods.", file=sys.stderr)
        return 1

    if not by_method:
        print("[FAIL] no results files passed guards", file=sys.stderr)
        return 1

    best_primary = max(e["primary"]["mean_precision_at_5"] for e in by_method.values())
    print(render_markdown_table(by_method, best_primary))

    # Update RESULTS.md
    for name, eval_ in by_method.items():
        row = render_results_md_row(name, eval_)
        append_or_replace_results_md(args.results_md, name, row)
    print(f"\nLeaderboard updated: {args.results_md}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
