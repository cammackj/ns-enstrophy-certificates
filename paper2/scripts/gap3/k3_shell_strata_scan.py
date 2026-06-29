#!/usr/bin/env python3
"""Scan k=3 shell strata to identify supports needing exact exclusion.

This is a finite-stratum triage tool for the full-block global certificate.  It
uses the complete current triad filter and runs a small deterministic multi-start
polish on selected shell supports.  The output is not a theorem; it is the map
of which strata require paper-grade interval/KKT certificates.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.gap3.max_b_over_keff import neg_ratio_and_grad  # noqa: E402
from scripts.gap3.multi_mode_beta_bound import divfree_basis, get_wavevectors, precompute_triads  # noqa: E402
from scripts.gap3.k3_closed_form_probe import support_from_warm  # noqa: E402


C3_TARGET = 0.021936469459403747249299192478957700397867315103825


def resolve_workers(requested: int, task_count: int) -> int:
    if requested == 0:
        return min(max(1, os.cpu_count() or 1), max(1, task_count))
    return min(max(1, requested), max(1, task_count))


def build_shell_problem(shells: tuple[int, ...]) -> dict:
    shell_set = set(shells)
    wavevectors = [wavevector for wavevector in get_wavevectors(max_shell2=15, min_shell2=8) if int(np.dot(wavevector, wavevector)) in shell_set]
    e1s = []
    e2s = []
    k2s = []
    for wavevector in wavevectors:
        e1, e2 = divfree_basis(wavevector)
        e1s.append(e1)
        e2s.append(e2)
        k2s.append(float(np.dot(wavevector, wavevector)))
    _, ell_idx, ell2, r_idx, s_idx, s_mat = precompute_triads(wavevectors)
    return {
        "shells": tuple(int(shell) for shell in shells),
        "wavevecs": [tuple(int(component) for component in wavevector) for wavevector in wavevectors],
        "N": len(wavevectors),
        "e1s": np.asarray(e1s, dtype=float),
        "e2s": np.asarray(e2s, dtype=float),
        "k2s": np.asarray(k2s, dtype=float),
        "ell_idx": ell_idx,
        "ell2": ell2,
        "r_idx": r_idx,
        "s_idx": s_idx,
        "s_mat": s_mat,
    }


def objective(problem: dict, params: np.ndarray) -> tuple[float, np.ndarray]:
    value, gradient = neg_ratio_and_grad(
        params,
        problem["N"],
        problem["e1s"],
        problem["e2s"],
        problem["k2s"],
        problem["ell_idx"],
        problem["ell2"],
        problem["r_idx"],
        problem["s_idx"],
        problem["s_mat"],
    )
    return value, gradient


def random_params(rng: np.random.Generator, modes: int) -> np.ndarray:
    params = np.empty(4 * modes, dtype=float)
    params[0::4] = rng.uniform(0.0, np.pi / 2.0, size=modes)
    params[1::4] = rng.uniform(0.0, 2.0 * np.pi, size=modes)
    params[2::4] = rng.uniform(0.0, 2.0 * np.pi, size=modes)
    params[3::4] = rng.normal(0.0, 1.0, size=modes)
    return params


def active_warm_start(problem: dict) -> np.ndarray | None:
    _, support_modes, support_params = support_from_warm()
    problem_index = {tuple(mode): index for index, mode in enumerate(problem["wavevecs"])}
    if any(tuple(mode) not in problem_index for mode in support_modes):
        return None
    params = np.empty(4 * problem["N"], dtype=float)
    params[0::4] = np.pi / 4.0
    params[1::4] = 0.0
    params[2::4] = 0.0
    params[3::4] = -32.0
    for support_index, mode in enumerate(support_modes):
        problem_mode_index = problem_index[tuple(mode)]
        params[4 * problem_mode_index : 4 * problem_mode_index + 4] = support_params[4 * support_index : 4 * support_index + 4]
    return params


def polish(problem: dict, start: np.ndarray, maxiter: int) -> tuple[float, float, np.ndarray]:
    bounds = [(0.0, np.pi / 2.0), (0.0, 2.0 * np.pi), (0.0, 2.0 * np.pi), (-40.0, 40.0)] * problem["N"]

    def fun(params: np.ndarray) -> float:
        return float(objective(problem, params)[0])

    def jac(params: np.ndarray) -> np.ndarray:
        return np.asarray(objective(problem, params)[1], dtype=float)

    result = minimize(fun, start, jac=jac, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter, "gtol": 1e-10, "ftol": 1e-14})
    value = -float(result.fun)
    grad_max = float(np.max(np.abs(result.jac))) if result.jac is not None else float("nan")
    return value, grad_max, np.asarray(result.x, dtype=float)


def active_mode_count(params: np.ndarray, floor: float = -20.0) -> int:
    return int(np.sum(params[3::4] > floor))


def scan_shells(shells: tuple[int, ...], starts: int, seed: int, maxiter: int, save_params: bool = False) -> dict:
    problem = build_shell_problem(shells)
    if len(problem["ell_idx"]) == 0:
        row = {
            "shells": shells,
            "modes": problem["N"],
            "triads": 0,
            "best_value": 0.0,
            "gap_to_target": C3_TARGET,
            "best_grad_max": 0.0,
            "active_modes": 0,
        }
        if save_params:
            row["wavevectors"] = problem["wavevecs"]
            row["best_params"] = []
        return row
    rng = np.random.default_rng(seed)
    best_value = -float("inf")
    best_grad = float("inf")
    best_params = None
    start_vectors = []
    warm = active_warm_start(problem)
    if warm is not None:
        start_vectors.append(warm)
    for _ in range(starts):
        start_vectors.append(random_params(rng, problem["N"]))
    for start in start_vectors:
        value, grad_max, params = polish(problem, start, maxiter)
        if value > best_value:
            best_value = value
            best_grad = grad_max
            best_params = params
    row = {
        "shells": shells,
        "modes": problem["N"],
        "triads": int(len(problem["ell_idx"])),
        "best_value": best_value,
        "gap_to_target": C3_TARGET - best_value,
        "best_grad_max": best_grad,
        "active_modes": active_mode_count(best_params) if best_params is not None else 0,
        "used_active_warm_start": warm is not None,
    }
    if save_params:
        row["wavevectors"] = problem["wavevecs"]
        row["best_params"] = best_params.tolist() if best_params is not None else []
    return row


def scan_task(task: tuple[int, tuple[int, ...], int, int, int, bool]) -> tuple[int, dict]:
    index, shells, starts, seed, maxiter, save_params = task
    return index, scan_shells(shells, starts, seed + index, maxiter, save_params)


def triad_connected_shell_subsets(max_size: int) -> list[tuple[int, ...]]:
    full = build_shell_problem((8, 9, 10, 11, 12, 13, 14))
    shells = sorted(set(int(shell) for shell in full["k2s"]))
    connected = []
    for size in range(1, max_size + 1):
        for subset in itertools.combinations(shells, size):
            if len(build_shell_problem(subset)["ell_idx"]) > 0:
                connected.append(subset)
    return connected


def parse_shells(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--maxiter", type=int, default=2000)
    parser.add_argument("--max-size", type=int, default=3)
    parser.add_argument("--workers", type=int, default=0, help="parallel shell-support workers; 0 uses all logical CPUs")
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--save-params", action="store_true", help="store best optimizer coordinates for later KKT certification")
    parser.add_argument("--shells", action="append", default=None, help="Explicit comma-separated shell support; may be repeated")
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    start_time = time.time()
    supports = [parse_shells(item) for item in args.shells] if args.shells else triad_connected_shell_subsets(args.max_size)
    workers = resolve_workers(args.workers, len(supports))
    print(f"k=3 shell-stratum scan: supports={len(supports)} starts={args.starts} max-size={args.max_size} workers={workers}")
    rows = []
    tasks = [(index, shells, args.starts, args.seed, args.maxiter, args.save_params) for index, shells in enumerate(supports, start=1)]
    if workers == 1:
        iterator = (scan_task(task) for task in tasks)
        for index, row in iterator:
            rows.append(row)
            print(
                f"{index:3d}/{len(supports)} shells={tuple(row['shells'])} modes={row['modes']:2d} triads={row['triads']:4d} "
                f"best={row['best_value']:.12f} gap={row['gap_to_target']:+.3e} grad={row['best_grad_max']:.2e} active={row['active_modes']}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(scan_task, task) for task in tasks]
            completed = 0
            for future in as_completed(futures):
                index, row = future.result()
                completed += 1
                rows.append(row)
                print(
                    f"{completed:3d}/{len(supports)} task={index:3d} shells={tuple(row['shells'])} modes={row['modes']:2d} triads={row['triads']:4d} "
                    f"best={row['best_value']:.12f} gap={row['gap_to_target']:+.3e} grad={row['best_grad_max']:.2e} active={row['active_modes']}",
                    flush=True,
                )
    rows.sort(key=lambda item: item["best_value"], reverse=True)
    print("\nTop shell strata")
    for row in rows[:10]:
        print(
            f"  shells={tuple(row['shells'])} best={row['best_value']:.15f} "
            f"gap={row['gap_to_target']:+.6e} active={row['active_modes']} modes={row['modes']}",
            flush=True,
        )
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "target": C3_TARGET,
        "starts": args.starts,
        "seed": args.seed,
        "maxiter": args.maxiter,
        "max_size": args.max_size,
        "workers": workers,
        "save_params": bool(args.save_params),
        "rows": rows,
        "elapsed_seconds": time.time() - start_time,
    }
    if not args.no_save:
        output = Path(args.output) if args.output else Path("results") / f"k3_shell_strata_scan_{datetime.now():%Y%m%d_%H%M%S}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved: {output}")


if __name__ == "__main__":
    main()