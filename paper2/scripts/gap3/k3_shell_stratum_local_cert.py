#!/usr/bin/env python3
"""Local certificate for a k=3 shell-stratum optimizer row.

Given a row saved by ``k3_shell_strata_scan.py --save-params``, this script
reconstructs the exact shell problem, evaluates the optimizer at high precision,
checks the projected gradient, and computes a parallel central-difference
Hessian of ``-R`` using the analytic gradient.  A non-DCA stratum with a clean
positive semidefinite Hessian and a large value gap is a finite KKT brick for
the global k=3 exclusion program.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.gap3.certify_k3_maximum import _build_mpmath_data, _mpmath_objective  # noqa: E402
from scripts.gap3.k3_shell_strata_scan import C3_TARGET, build_shell_problem, objective  # noqa: E402


_WORKER_PROBLEM: dict[str, Any] | None = None
_WORKER_PARAMS: np.ndarray | None = None
_WORKER_STEP: float | None = None


def resolve_workers(requested: int, task_count: int) -> int:
    if requested == 0:
        return min(max(1, os.cpu_count() or 1), max(1, task_count))
    return min(max(1, requested), max(1, task_count))


def init_hessian_worker(problem: dict[str, Any], params: list[float], step: float) -> None:
    global _WORKER_PROBLEM, _WORKER_PARAMS, _WORKER_STEP
    _WORKER_PROBLEM = problem
    _WORKER_PARAMS = np.asarray(params, dtype=float)
    _WORKER_STEP = float(step)


def hessian_column(index: int) -> tuple[int, list[float]]:
    if _WORKER_PROBLEM is None or _WORKER_PARAMS is None or _WORKER_STEP is None:
        raise RuntimeError("hessian worker is not initialized")
    plus = _WORKER_PARAMS.copy()
    minus = _WORKER_PARAMS.copy()
    plus[index] += _WORKER_STEP
    minus[index] -= _WORKER_STEP
    _, grad_plus = objective(_WORKER_PROBLEM, plus)
    _, grad_minus = objective(_WORKER_PROBLEM, minus)
    column = (np.asarray(grad_plus, dtype=float) - np.asarray(grad_minus, dtype=float)) / (2.0 * _WORKER_STEP)
    return index, column.tolist()


def choose_row(data: dict[str, Any], rank: int, require_non_dca: bool) -> tuple[int, dict[str, Any]]:
    dca = {8, 10, 14}
    rows = data["rows"]
    filtered = []
    for index, row in enumerate(rows):
        if require_non_dca and dca.issubset(set(int(shell) for shell in row["shells"])):
            continue
        filtered.append((index, row))
    if not filtered:
        raise RuntimeError("no row matched the requested filter")
    if rank < 0 or rank >= len(filtered):
        raise ValueError(f"rank must be in [0,{len(filtered)-1}]")
    return filtered[rank]


def mpmath_value(problem: dict[str, Any], params: np.ndarray, dps: int) -> str:
    mp.mp.dps = dps
    params_mp = [mp.mpf(repr(float(value))) for value in params]
    value = _mpmath_objective(params_mp, *_build_mpmath_data(problem, dps))
    return mp.nstr(value, n=dps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strata-json", required=True)
    parser.add_argument("--rank", type=int, default=0, help="rank within the selected row filter")
    parser.add_argument("--require-non-dca", action="store_true")
    parser.add_argument("--dps", type=int, default=60)
    parser.add_argument("--hessian-step", type=float, default=1e-5)
    parser.add_argument("--flat-tol", type=float, default=1e-7)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    start_time = time.time()
    data = json.loads(Path(args.strata_json).read_text(encoding="utf-8"))
    source_index, row = choose_row(data, args.rank, args.require_non_dca)
    if "best_params" not in row or not row["best_params"]:
        raise RuntimeError("selected row has no best_params; rerun shell scan with --save-params")
    shells = tuple(int(shell) for shell in row["shells"])
    problem = build_shell_problem(shells)
    params = np.asarray(row["best_params"], dtype=float)
    if params.shape != (4 * int(problem["N"]),):
        raise RuntimeError(f"parameter length {len(params)} does not match shell problem dimension {4 * int(problem['N'])}")

    neg_value, gradient = objective(problem, params)
    value = -float(neg_value)
    gradient_max = float(np.max(np.abs(gradient)))
    value_mp = mpmath_value(problem, params, args.dps)

    dimension = len(params)
    workers = resolve_workers(args.workers, dimension)
    print("k=3 shell-stratum local certificate")
    print("====================================")
    print(f"source_row={source_index} shells={shells} modes={problem['N']} triads={len(problem['ell_idx'])}")
    print(f"value={value:.15f} gap={C3_TARGET - value:+.6e} grad={gradient_max:.3e}")
    print(f"hessian dimension={dimension} workers={workers} step={args.hessian_step:g}", flush=True)

    hessian = np.zeros((dimension, dimension), dtype=float)
    if workers == 1:
        init_hessian_worker(problem, params.tolist(), args.hessian_step)
        for index in range(dimension):
            _, column = hessian_column(index)
            hessian[:, index] = column
            if (index + 1) % 20 == 0 or index + 1 == dimension:
                print(f"  hessian {index + 1}/{dimension}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=init_hessian_worker, initargs=(problem, params.tolist(), args.hessian_step)) as executor:
            futures = [executor.submit(hessian_column, index) for index in range(dimension)]
            completed = 0
            for future in as_completed(futures):
                index, column = future.result()
                completed += 1
                hessian[:, index] = column
                if completed % 20 == 0 or completed == dimension:
                    print(f"  hessian {completed}/{dimension}", flush=True)

    hessian = 0.5 * (hessian + hessian.T)
    eigenvalues = np.linalg.eigvalsh(hessian)
    negative = int(np.sum(eigenvalues < -args.flat_tol))
    flat = int(np.sum(np.abs(eigenvalues) <= args.flat_tol))
    positive = int(np.sum(eigenvalues > args.flat_tol))
    print(f"H(-R): min={eigenvalues[0]:+.6e} max={eigenvalues[-1]:+.6e} counts={negative} neg / {flat} flat / {positive} pos")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "shell_stratum_local_cert",
        "source_json": args.strata_json,
        "source_row_index": int(source_index),
        "shells": list(shells),
        "target": C3_TARGET,
        "value_float": value,
        "value_mpmath": value_mp,
        "gap_to_target_float": C3_TARGET - value,
        "gradient_max_abs": gradient_max,
        "modes": int(problem["N"]),
        "triads": int(len(problem["ell_idx"])),
        "hessian_step": args.hessian_step,
        "flat_tol": args.flat_tol,
        "workers": workers,
        "hessian_min_eigenvalue": float(eigenvalues[0]),
        "hessian_max_eigenvalue": float(eigenvalues[-1]),
        "hessian_counts": {"negative": negative, "flat": flat, "positive": positive},
        "smallest_eigenvalues": [float(value) for value in eigenvalues[:20]],
        "largest_eigenvalues": [float(value) for value in eigenvalues[-20:]],
        "elapsed_seconds": time.time() - start_time,
        "method": "mpmath value plus parallel central-difference Hessian of -R using analytic gradient",
    }
    if not args.no_save:
        output = Path(args.output) if args.output else Path("results") / f"k3_shell_stratum_local_cert_{datetime.now():%Y%m%d_%H%M%S}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved: {output}")


if __name__ == "__main__":
    main()