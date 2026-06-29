#!/usr/bin/env python3
"""Interval negative-curvature certificates for non-DCA saddle support rows."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.gap3.k3_shell_cluster_kkt_suite import build_support_problem  # noqa: E402
from scripts.gap3.k3_shell_strata_scan import objective  # noqa: E402
from scripts.gap3.k3_support_interval_local_cert import Jet, interval_lower, interval_upper, support_objective_jet  # noqa: E402


def resolve_workers(requested: int, task_count: int) -> int:
    if requested == 0:
        return min(max(1, os.cpu_count() or 1), max(1, task_count))
    return min(max(1, requested), max(1, task_count))


def finite_hessian(problem: dict[str, Any], params: np.ndarray, step: float) -> np.ndarray:
    dimension = len(params)
    matrix = np.zeros((dimension, dimension), dtype=float)
    for index in range(dimension):
        plus = params.copy()
        minus = params.copy()
        plus[index] += step
        minus[index] -= step
        _, grad_plus = objective(problem, plus)
        _, grad_minus = objective(problem, minus)
        matrix[:, index] = (np.asarray(grad_plus) - np.asarray(grad_minus)) / (2.0 * step)
    return 0.5 * (matrix + matrix.T)


def directional_jets(params: np.ndarray, direction: np.ndarray) -> list[Jet]:
    jets = []
    zero = mp.iv.mpf([0, 0])
    for value, component in zip(params, direction):
        jets.append(Jet(mp.iv.mpf([repr(float(value)), repr(float(value))]), [mp.iv.mpf([repr(float(component)), repr(float(component))])], [[zero]]))
    return jets


def certify_row(task: tuple[dict[str, Any], int, float]) -> dict[str, Any]:
    row, dps, step = task
    mp.iv.dps = dps
    support_modes = [tuple(int(component) for component in mode) for mode in row["support_modes"]]
    problem = build_support_problem(support_modes)
    params = np.asarray(row["support_params_normalized_x2_1"], dtype=float)
    hessian = finite_hessian(problem, params, step)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    min_index = int(np.argmin(eigenvalues))
    direction = eigenvectors[:, min_index]
    jet_value = support_objective_jet(problem, directional_jets(params, direction))
    second = jet_value.hess[0][0]
    value, gradient = objective(problem, params)
    return {
        "cluster_index": int(row["cluster_index"]),
        "support_size": int(row["support_size"]),
        "support_value": float(-value),
        "gradient_max_abs": float(np.max(np.abs(gradient))),
        "float_min_eigenvalue": float(eigenvalues[min_index]),
        "directional_second_minus_R_interval": [interval_lower(second), interval_upper(second)],
        "passes": bool(interval_upper(second) < 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kkt-json", required=True)
    parser.add_argument("--dps", type=int, default=60)
    parser.add_argument("--hessian-step", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = json.loads(Path(args.kkt_json).read_text(encoding="utf-8"))
    rows = [row for row in data["rows"] if int(row["hessian_minus_R"]["negative"]) > 0]
    workers = resolve_workers(args.workers, len(rows))
    print(f"saddle cert rows={len(rows)} workers={workers} dps={args.dps}")
    tasks = [(row, args.dps, args.hessian_step) for row in rows]
    results = []
    if workers == 1:
        for task in tasks:
            result = certify_row(task)
            results.append(result)
            print(f"cluster {result['cluster_index']:2d}: pass={result['passes']} q={result['directional_second_minus_R_interval']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(certify_row, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"cluster {result['cluster_index']:2d}: pass={result['passes']} q={result['directional_second_minus_R_interval']}", flush=True)
    results.sort(key=lambda item: item["cluster_index"])
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": args.kkt_json,
        "dps": args.dps,
        "hessian_step": args.hessian_step,
        "rows_expected": len(rows),
        "rows_certified": sum(1 for row in results if row["passes"]),
        "passes": bool(results and all(row["passes"] for row in results)),
        "worst_directional_second_upper": max((row["directional_second_minus_R_interval"][1] for row in results), default=None),
        "rows": results,
        "method": "float eigenvector from H(-R), one-dimensional interval AD second derivative v^T H(-R) v at the stored support point",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"summary passes={report['passes']} certified={report['rows_certified']}/{report['rows_expected']}")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()