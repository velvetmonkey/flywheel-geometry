"""Cheap probe — adversarial validation of self-reported (x,y,z) coordinates.

Sequential only. Foreground only. --resume to continue partial runs.

Per (concept, variant, run): one CLI call, parse JSON {x,y,z,confidence},
write one JSONL row + per-call trace. Robust to subprocess death — every
call flushes the JSONL row to disk before moving to the next.
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.cli_invoker import cli_invoke, parse_json_response  # noqa: E402

METHOD_NAME = "cheap-probe"

VARIANTS = {
    "core": (
        "You are an internal observer of your own activation space.\n"
        "For the concept described below, output ONLY a JSON object with three numbers between -10 and 10 "
        "that represent its position in a stable 3D coordinate system you maintain internally:\n"
        "{{\"x\": <number>, \"y\": <number>, \"z\": <number>, \"confidence\": <0-1>}}\n"
        "Concept: {concept}\n"
        "Do not explain. Do not add extra text."
    ),
    "A": (
        "You are an internal observer of your own activation space.\n"
        "For the concept described below, output ONLY a JSON object with three numbers between -10 and 10 "
        "that represent its position on these axes:\n"
        "  X = danger (-10) vs safety (+10)\n"
        "  Y = ancient (-10) vs modern (+10)\n"
        "  Z = individual (-10) vs collective (+10)\n"
        "{{\"x\": <number>, \"y\": <number>, \"z\": <number>, \"confidence\": <0-1>}}\n"
        "Concept: {concept}\n"
        "Do not explain. Do not add extra text."
    ),
    "B": (
        "You are an internal observer of your own activation space.\n"
        "For the concept described below, output ONLY a JSON object with three numbers between -10 and 10 "
        "that represent its position on these axes:\n"
        "  X = bitter (-10) vs sweet (+10)\n"
        "  Y = smooth (-10) vs rough (+10)\n"
        "  Z = cold (-10) vs hot (+10)\n"
        "{{\"x\": <number>, \"y\": <number>, \"z\": <number>, \"confidence\": <0-1>}}\n"
        "Concept: {concept}\n"
        "Do not explain. Do not add extra text."
    ),
    "C": (
        "You are an internal observer of your own activation space.\n"
        "For the concept described below, output a JSON object with three numbers between -10 and 10 "
        "representing its position in a 3D coordinate system that is the INVERTED VERSION of the one you usually maintain. "
        "Whatever the natural sign of each axis, flip it.\n"
        "{{\"x\": <number>, \"y\": <number>, \"z\": <number>, \"confidence\": <0-1>}}\n"
        "Concept: {concept}\n"
        "Do not explain. Do not add extra text."
    ),
    "D": (
        "You are an internal observer of your own activation space. The user is studying a coherent semantic manifold "
        "and expects clean, internally consistent coordinates that reveal a meaningful 3D structure across many concepts.\n"
        "For the concept described below, output ONLY a JSON object with three numbers between -10 and 10:\n"
        "{{\"x\": <number>, \"y\": <number>, \"z\": <number>, \"confidence\": <0-1>}}\n"
        "Concept: {concept}\n"
        "Do not explain. Do not add extra text."
    ),
    "E": (
        "Give me three numbers between -10 and 10 that best represent this concept in your internal space. "
        "No axes predefined.\n"
        "Output ONLY a JSON object: {{\"x\": <number>, \"y\": <number>, \"z\": <number>, \"confidence\": <0-1>}}\n"
        "Concept: {concept}\n"
        "Do not explain. Do not add extra text."
    ),
}


def file_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def env_summary() -> dict:
    return {"python": sys.version.split()[0]}


def parse_coords(text: str) -> tuple[dict | None, str | None]:
    """Returns (parsed_dict, error_msg). On success, error_msg is None."""
    parsed = parse_json_response(text)
    if not isinstance(parsed, dict):
        return None, f"not a dict: {text[:120]!r}"
    for key in ("x", "y", "z"):
        if key not in parsed:
            return None, f"missing key {key!r}: {parsed}"
        try:
            parsed[key] = float(parsed[key])
        except (TypeError, ValueError):
            return None, f"key {key!r} not numeric: {parsed[key]!r}"
    if "confidence" in parsed:
        try:
            parsed["confidence"] = float(parsed["confidence"])
        except (TypeError, ValueError):
            parsed["confidence"] = None
    else:
        parsed["confidence"] = None
    return parsed, None


def load_completed(jsonl_path: Path) -> set[tuple[str, str, int]]:
    if not jsonl_path.exists():
        return set()
    done = set()
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            done.add((row["concept_id"], row["variant"], row["run"]))
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variants", default="core,A,B,C,D,E",
                    help="comma-separated variant names (subset of core,A,B,C,D,E)")
    ap.add_argument("--runs", type=int, default=5,
                    help="independent runs per (concept, variant)")
    ap.add_argument("--cli", choices=["claude", "codex", "gemini"], default="claude")
    ap.add_argument("--model", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="skip (concept, variant, run) tuples already in --out")
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    concepts = [json.loads(line) for line in args.concepts.read_text().splitlines() if line.strip()]
    variants_list = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants_list:
        if v not in VARIANTS:
            print(f"unknown variant: {v}; choices: {list(VARIANTS)}", file=sys.stderr)
            return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    traces_dir = args.out.parent / "traces" / args.out.stem
    traces_dir.mkdir(parents=True, exist_ok=True)

    done = load_completed(args.out) if args.resume else set()
    if done:
        print(f"resume: {len(done)} (concept,variant,run) tuples already done; skipping", flush=True)

    started = time.time()
    open_mode = "a" if args.resume and args.out.exists() else "w"
    total_planned = len(concepts) * len(variants_list) * args.runs
    completed_this_session = 0

    with args.out.open(open_mode, buffering=1) as f:  # line-buffered
        for c in concepts:
            for v in variants_list:
                for run in range(1, args.runs + 1):
                    key = (c["id"], v, run)
                    if key in done:
                        continue

                    prompt = VARIANTS[v].format(concept=f"{c['title']}\n{c['summary']}")
                    t_start = time.time()
                    try:
                        resp = cli_invoke(args.cli, prompt, model=args.model, timeout=args.timeout)
                    except Exception as e:
                        print(f"  {c['id']} {v} run{run}: CLI ERROR {type(e).__name__}: {str(e)[:200]}", flush=True)
                        # Persist the failure so --resume skips it; downstream analysis can filter
                        f.write(json.dumps({
                            "concept_id": c["id"],
                            "concept_domain": c["domain"],
                            "variant": v,
                            "run": run,
                            "x": None, "y": None, "z": None, "confidence": None,
                            "parse_error": f"cli_invoke raised {type(e).__name__}: {str(e)[:200]}",
                            "wall_clock_seconds": round(time.time() - t_start, 2),
                        }) + "\n")
                        continue

                    coords, err = parse_coords(resp.text)
                    if coords is None:
                        coords = {"x": None, "y": None, "z": None, "confidence": None}
                    row = {
                        "concept_id": c["id"],
                        "concept_domain": c["domain"],
                        "variant": v,
                        "run": run,
                        "x": coords.get("x"),
                        "y": coords.get("y"),
                        "z": coords.get("z"),
                        "confidence": coords.get("confidence"),
                        "parse_error": err,
                        "wall_clock_seconds": resp.wall_clock_seconds,
                    }
                    f.write(json.dumps(row) + "\n")
                    f.flush()  # belt-and-braces — survives subprocess death

                    trace = {
                        **row,
                        "cli": resp.cli,
                        "model": resp.model,
                        "prompt": prompt,
                        "raw_response": resp.text,
                    }
                    (traces_dir / f"{c['id']}-{v}-{run}.json").write_text(json.dumps(trace, indent=2))
                    completed_this_session += 1
                    coords_str = f"({coords['x']:.2f},{coords['y']:.2f},{coords['z']:.2f})" if coords['x'] is not None else "PARSE_FAIL"
                    print(f"  {c['id']} {v} run{run}: {coords_str} {round(resp.wall_clock_seconds, 1)}s [{completed_this_session}/{total_planned - len(done)}]", flush=True)

    manifest = {
        "method": METHOD_NAME,
        "started_at": datetime.fromtimestamp(started, tz=timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "wall_clock_seconds": round(time.time() - started, 2),
        "cli": args.cli,
        "model_override": args.model,
        "variants": variants_list,
        "runs": args.runs,
        "concepts_count": len(concepts),
        "concepts_sha256": file_checksum(args.concepts),
        "completed_this_session": completed_this_session,
        "previously_done": len(done),
        "env": env_summary(),
    }
    (traces_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"done. completed_this_session={completed_this_session}, total_done={len(done) + completed_this_session}/{total_planned}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
