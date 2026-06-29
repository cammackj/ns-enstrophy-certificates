#!/usr/bin/env python3
"""High-precision replay for k=3 support KKT ledger rows."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import mpmath as mp

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.gap3.certify_k3_maximum import _build_mpmath_data, _mpmath_objective  # noqa: E402
from scripts.gap3.k3_shell_cluster_kkt_suite import build_support_problem  # noqa: E402


C3_TARGET_TEXT = "0.021936469459403747249299192478957700397867315103825"


def resolve_workers(requested: int, task_count: int) -> int:
    if requested == 0:
        return min(max(1, os.cpu_count() or 1), max(1, task_count))
    return min(max(1, requested), max(1, task_count))


def parse_cluster_indices(text: str) -> set[int] | None:
    if not text.strip() or text.strip().lower() == "all":
        return None
    return {int(part) for part in text.split(",") if part.strip()}


def replay_row(task: tuple[dict[str, Any], int]) -> dict[str, Any]:
    row, dps = task
    mp.mp.dps = dps
    support_modes = [tuple(int(component) for component in mode) for mode in row["support_modes"]]
    problem = build_support_problem(support_modes)
    params = [mp.mpf(repr(float(value))) for value in row["support_params_normalized_x2_1"]]
    value = _mpmath_objective(params, *_build_mpmath_data(problem, dps))
    gap = mp.mpf(C3_TARGET_TEXT) - value
    return {
        "cluster_index": int(row["cluster_index"]),
        "support_size": int(row["support_size"]),
        "hessian_negative": int(row["hessian_minus_R"]["negative"]),
        "support_value_float": float(row["support_value"]),
        "support_value_mpmath": mp.nstr(value, n=dps),
        "gap_to_target_mpmath": mp.nstr(gap, n=dps),
        "float_minus_mpmath": mp.nstr(mp.mpf(repr(float(row["support_value"]))) - value, n=20),
        "gap_to_target_float": float(row["support_gap_to_target"]),
        "minimum_release": float(row["full_release"]["minimum"]["min_release_coefficient"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kkt-json", required=True)
    parser.add_argument("--clusters", default="all", help="comma-separated cluster indices or all")
    parser.add_argument("--kkt-only", action="store_true", help="only replay rows with no negative support Hessian directions")
    parser.add_argument("--dps", type=int, default=80)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = json.loads(Path(args.kkt_json).read_text(encoding="utf-8"))
    selected = parse_cluster_indices(args.clusters)
    rows = []
    for row in data["rows"]:
        if selected is not None and int(row["cluster_index"]) not in selected:
            continue
        if args.kkt_only and int(row["hessian_minus_R"]["negative"]):
            continue
        rows.append(row)
    workers = resolve_workers(args.workers, len(rows))
    print(f"mp replay rows={len(rows)} workers={workers} dps={args.dps}")
    tasks = [(row, args.dps) for row in rows]
    results = []
    if workers == 1:
        for task in tasks:
            result = replay_row(task)
            results.append(result)
            print(f"cluster {result['cluster_index']:2d}: {result['support_value_mpmath']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(replay_row, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"cluster {result['cluster_index']:2d}: {result['support_value_mpmath']}", flush=True)
    results.sort(key=lambda item: item["support_value_float"], reverse=True)
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": args.kkt_json,
        "target_float": data["target"],
        "target_mpmath_text": C3_TARGET_TEXT,
        "dps": args.dps,
        "kkt_only": bool(args.kkt_only),
        "rows_replayed": len(results),
        "best_replayed": results[0] if results else None,
        "rows": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if results:
        print(f"best replayed cluster={results[0]['cluster_index']} value={results[0]['support_value_mpmath']}")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()