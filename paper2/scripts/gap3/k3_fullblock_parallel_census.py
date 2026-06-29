#!/usr/bin/env python3
"""Parallel full-block k=3 basin census at a low inactive floor.

This is not an algebraic proof.  It is a deliberately heavy numerical census
for the full 85-mode k=3 block using the current complete triad filter and a
low log-amplitude floor.  It exists to answer the practical question: if we
push the entire shell hard in parallel, do we find any basin above the refined
9-mode algebraic candidate or above the penalized target?

Two scan modes are supported:

* raw: maximize R(u)=B/(X^2 D)
* penalized: maximize R(u)+mu*E_perp(u), where E_perp is the X-energy fraction
  outside the 9 active modes

The output JSON is a basin ledger, not a manuscript certificate.  It is useful
for finding counterexamples and for prioritizing the exact certificate route.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.gap3.k3_active_set_verify import build_problem_scope  # noqa: E402
from scripts.gap3.k3_penalized_fullblock_scan import (  # noqa: E402
    C3_TARGET,
    active_indices,
    active_warm_start,
    e_perp_and_grad,
)
from scripts.gap3.max_b_over_keff import neg_ratio_and_grad  # noqa: E402


_WORKER: dict[str, Any] = {}


def parse_start_mix(text: str) -> dict[str, int]:
    mix: dict[str, int] = {}
    if not text.strip():
        return mix
    for part in text.split(","):
        if not part.strip():
            continue
        name, count_text = part.split(":", 1)
        count = int(count_text)
        if count < 0:
            raise ValueError("start counts must be nonnegative")
        mix[name.strip()] = count
    return mix


def parse_task_seeds(text: str) -> list[tuple[int, str, int]]:
    tasks: list[tuple[int, str, int]] = []
    if not text.strip():
        return tasks
    for part in text.split(","):
        if not part.strip():
            continue
        kind, seed_text = part.split(":", 1)
        tasks.append((len(tasks) + 1, kind.strip(), int(seed_text)))
    return tasks


def worker_count(requested: int) -> int:
    if requested == 0:
        return max(1, os.cpu_count() or 1)
    return max(1, requested)


def make_bounds(n_modes: int, loga_floor: float, loga_ceiling: float) -> list[tuple[float, float]]:
    return [(0.0, math.pi / 2.0), (0.0, 2.0 * math.pi), (0.0, 2.0 * math.pi), (loga_floor, loga_ceiling)] * n_modes


def positive_rep(mode: tuple[int, int, int]) -> tuple[int, int, int]:
    neg = tuple(-component for component in mode)
    return mode if mode > neg else neg


def build_orbit_support_indices(problem: dict) -> list[list[int]]:
    mode_to_index = {tuple(int(component) for component in mode): index for index, mode in enumerate(problem["wavevecs"])}
    canonical = [tuple(int(component) for component in problem["wavevecs"][index]) for index in active_indices(problem)]
    supports: set[tuple[int, ...]] = set()
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            indices = []
            for mode in canonical:
                mapped = tuple(signs[index] * mode[perm[index]] for index in range(3))
                indices.append(mode_to_index[positive_rep(mapped)])
            supports.add(tuple(sorted(indices)))
    return [list(support) for support in sorted(supports)]


def orbit_inactive_masks(problem: dict, orbit_supports: list[list[int]]) -> list[np.ndarray]:
    masks = []
    for support in orbit_supports:
        mask = np.ones(problem["N"], dtype=bool)
        mask[np.asarray(support, dtype=int)] = False
        masks.append(mask)
    return masks


def init_worker(problem: dict, inactive_mask: np.ndarray, orbit_masks: list[np.ndarray], options: dict) -> None:
    _WORKER.clear()
    _WORKER["problem"] = problem
    _WORKER["inactive_mask"] = inactive_mask
    _WORKER["orbit_inactive_masks"] = orbit_masks
    _WORKER["options"] = options
    _WORKER["bounds"] = make_bounds(problem["N"], options["loga_floor"], options["loga_ceiling"])
    _WORKER["active_seed"] = active_warm_start(problem)


def random_angles(rng: np.random.Generator, n_modes: int) -> np.ndarray:
    params = np.empty(4 * n_modes, dtype=float)
    params[0::4] = rng.uniform(0.0, math.pi / 2.0, size=n_modes)
    params[1::4] = rng.uniform(0.0, 2.0 * math.pi, size=n_modes)
    params[2::4] = rng.uniform(0.0, 2.0 * math.pi, size=n_modes)
    return params


def make_start(kind: str, seed: int) -> np.ndarray:
    problem = _WORKER["problem"]
    options = _WORKER["options"]
    rng = np.random.default_rng(seed)
    n_modes = int(problem["N"])
    params = random_angles(rng, n_modes)
    floor = float(options["loga_floor"])

    if kind == "active":
        return np.asarray(_WORKER["active_seed"], dtype=float).copy()

    if kind == "active-small":
        params = np.asarray(_WORKER["active_seed"], dtype=float).copy()
        active = np.asarray(options["active_indices"], dtype=int)
        for index in active:
            base = 4 * index
            params[base : base + 3] += rng.normal(0.0, 1e-3, size=3)
            params[base + 3] += rng.normal(0.0, 1e-3)
        params[0::4] = np.clip(params[0::4], 0.0, math.pi / 2.0)
        params[1::4] = np.mod(params[1::4], 2.0 * math.pi)
        params[2::4] = np.mod(params[2::4], 2.0 * math.pi)
        params[3::4] = np.maximum(params[3::4], floor)
        return params

    if kind == "active-wide":
        params = np.asarray(_WORKER["active_seed"], dtype=float).copy()
        active = np.asarray(options["active_indices"], dtype=int)
        for index in active:
            base = 4 * index
            params[base : base + 3] += rng.normal(0.0, 5e-2, size=3)
            params[base + 3] += rng.normal(0.0, 5e-2)
        inactive = np.ones(n_modes, dtype=bool)
        inactive[active] = False
        params[4 * np.flatnonzero(inactive) + 3] = floor + rng.exponential(1.0, size=int(np.sum(inactive)))
        params[0::4] = np.clip(params[0::4], 0.0, math.pi / 2.0)
        params[1::4] = np.mod(params[1::4], 2.0 * math.pi)
        params[2::4] = np.mod(params[2::4], 2.0 * math.pi)
        return params

    if kind == "dense-normal":
        params[3::4] = rng.normal(0.0, 1.5, size=n_modes)
        return params

    if kind == "dense-wide":
        params[3::4] = rng.uniform(-8.0, 8.0, size=n_modes)
        return params

    if kind == "sparse":
        params[3::4] = floor
        support_size = int(rng.integers(3, min(24, n_modes) + 1))
        support = rng.choice(n_modes, size=support_size, replace=False)
        params[4 * support + 3] = rng.normal(0.0, 1.0, size=support_size)
        return params

    if kind == "shell-sparse":
        params[3::4] = floor
        shells = sorted(set(int(shell) for shell in problem["k2s"]))
        shell_count = int(rng.integers(2, min(5, len(shells)) + 1))
        chosen = set(rng.choice(shells, size=shell_count, replace=False).tolist())
        support = [index for index, shell in enumerate(problem["k2s"].astype(int)) if int(shell) in chosen]
        params[4 * np.asarray(support, dtype=int) + 3] = rng.normal(0.0, 1.0, size=len(support))
        return params

    raise ValueError(f"unknown start kind {kind!r}")


def objective_and_grad(params: np.ndarray) -> tuple[float, np.ndarray, float, float]:
    problem = _WORKER["problem"]
    options = _WORKER["options"]
    inactive_mask = _WORKER["inactive_mask"]
    negative_r, grad_negative_r = neg_ratio_and_grad(
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
    r_value = -float(negative_r)
    e_perp, grad_e_perp = e_perp_and_grad(problem, params, inactive_mask)
    if options["mode"] == "raw":
        return negative_r, grad_negative_r, r_value, e_perp
    if options["mode"] == "orbit-penalized":
        best_e = math.inf
        best_grad = grad_e_perp
        for mask in _WORKER["orbit_inactive_masks"]:
            candidate_e, candidate_grad = e_perp_and_grad(problem, params, mask)
            if candidate_e < best_e:
                best_e = candidate_e
                best_grad = candidate_grad
        objective = negative_r - options["mu"] * best_e
        gradient = grad_negative_r - options["mu"] * best_grad
        return objective, gradient, r_value, best_e
    objective = negative_r - options["mu"] * e_perp
    gradient = grad_negative_r - options["mu"] * grad_e_perp
    return objective, gradient, r_value, e_perp


def support_summary(params: np.ndarray, threshold: float) -> dict:
    problem = _WORKER["problem"]
    active = np.flatnonzero(params[3::4] > threshold)
    shells = sorted(set(int(problem["k2s"][index]) for index in active))
    return {
        "threshold": threshold,
        "active_modes": int(len(active)),
        "shells": shells,
        "mode_indices": [int(index) for index in active],
        "wavevectors": [tuple(int(component) for component in problem["wavevecs"][index]) for index in active],
    }


def run_task(task: tuple[int, str, int]) -> dict:
    ordinal, kind, seed = task
    options = _WORKER["options"]
    start = make_start(kind, seed)

    def fun(params: np.ndarray) -> float:
        value, _, _, _ = objective_and_grad(params)
        return value

    def jac(params: np.ndarray) -> np.ndarray:
        _, gradient, _, _ = objective_and_grad(params)
        return gradient

    result = minimize(
        fun,
        start,
        method="L-BFGS-B",
        jac=jac,
        bounds=_WORKER["bounds"],
        options={"ftol": options["ftol"], "gtol": options["gtol"], "maxiter": options["maxiter"], "maxls": 50},
    )
    value, gradient, r_value, e_perp = objective_and_grad(np.asarray(result.x, dtype=float))
    maximized = -float(value)
    row = {
        "ordinal": ordinal,
        "kind": kind,
        "seed": seed,
        "maximized_value": maximized,
        "R": r_value,
        "E_perp": e_perp,
        "gap_to_target": C3_TARGET - maximized,
        "gradient_max_abs": float(np.max(np.abs(gradient))),
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "support_floor20": support_summary(result.x, -20.0),
        "support_floor30": support_summary(result.x, -30.0),
    }
    if options["save_params"]:
        row["params"] = np.asarray(result.x, dtype=float).tolist()
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("raw", "penalized", "orbit-penalized"), default="penalized")
    parser.add_argument("--mu", type=float, default=1e-5)
    parser.add_argument("--start-mix", default="active:1,active-small:64,active-wide:64,dense-normal:256,dense-wide:256,sparse:256,shell-sparse:256")
    parser.add_argument("--task-seeds", default="", help="Replay exact starts as kind:seed comma pairs; bypasses --start-mix.")
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--maxiter", type=int, default=1600)
    parser.add_argument("--gtol", type=float, default=1e-10)
    parser.add_argument("--ftol", type=float, default=1e-14)
    parser.add_argument("--loga-floor", type=float, default=-40.0)
    parser.add_argument("--loga-ceiling", type=float, default=40.0)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--save-params", action="store_true", help="Store final optimizer coordinates in the JSON ledger.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    start_time = time.time()
    problem = build_problem_scope("full-block")
    active = active_indices(problem)
    inactive_mask = np.ones(problem["N"], dtype=bool)
    inactive_mask[active] = False
    orbit_supports = build_orbit_support_indices(problem)
    orbit_masks = orbit_inactive_masks(problem, orbit_supports)
    options = {
        "mode": args.mode,
        "mu": args.mu,
        "maxiter": args.maxiter,
        "gtol": args.gtol,
        "ftol": args.ftol,
        "loga_floor": args.loga_floor,
        "loga_ceiling": args.loga_ceiling,
        "active_indices": active,
        "save_params": args.save_params,
    }
    task_seeds = parse_task_seeds(args.task_seeds)
    mix = {} if task_seeds else parse_start_mix(args.start_mix)
    tasks: list[tuple[int, str, int]] = task_seeds
    if not tasks:
        rng = np.random.default_rng(args.seed)
        for kind, count in mix.items():
            for _ in range(count):
                tasks.append((len(tasks) + 1, kind, int(rng.integers(0, 2**31 - 1))))

    workers = worker_count(args.workers)
    print("k=3 full-block parallel census")
    print("================================")
    print(f"mode={args.mode} mu={args.mu:g} target={C3_TARGET:.17g}")
    print(f"problem: N={problem['N']} T={len(problem['ell_idx'])} active={len(active)} inactive={int(np.sum(inactive_mask))} orbit_supports={len(orbit_supports)}")
    print(f"starts={len(tasks)} mix={mix} task_seeds={args.task_seeds or '-'} workers={workers} floor={args.loga_floor:g}")

    results = []
    best: dict | None = None
    if workers == 1:
        init_worker(problem, inactive_mask, orbit_masks, options)
        for task in tasks:
            row = run_task(task)
            results.append(row)
            if best is None or row["maximized_value"] > best["maximized_value"]:
                best = row
            if args.progress_every and len(results) % args.progress_every == 0:
                print(
                    f"{len(results):5d}/{len(tasks)} best={best['maximized_value']:.15f} "
                    f"gap={best['gap_to_target']:+.3e} kind={best['kind']} active20={best['support_floor20']['active_modes']}",
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(problem, inactive_mask, orbit_masks, options)) as executor:
            futures = [executor.submit(run_task, task) for task in tasks]
            for future in as_completed(futures):
                row = future.result()
                results.append(row)
                if best is None or row["maximized_value"] > best["maximized_value"]:
                    best = row
                if args.progress_every and len(results) % args.progress_every == 0:
                    print(
                        f"{len(results):5d}/{len(tasks)} best={best['maximized_value']:.15f} "
                        f"gap={best['gap_to_target']:+.3e} kind={best['kind']} active20={best['support_floor20']['active_modes']}",
                        flush=True,
                    )
    assert best is not None
    elapsed = time.time() - start_time
    results.sort(key=lambda row: row["maximized_value"], reverse=True)
    by_kind: dict[str, dict] = {}
    for row in results:
        entry = by_kind.setdefault(row["kind"], {"count": 0, "best": row})
        entry["count"] += 1
        if row["maximized_value"] > entry["best"]["maximized_value"]:
            entry["best"] = row

    print("\nSummary")
    print("-------")
    print(f"best={results[0]['maximized_value']:.17g} gap={results[0]['gap_to_target']:+.6e} kind={results[0]['kind']}")
    print(f"R={results[0]['R']:.17g} E_perp={results[0]['E_perp']:.6e} active20={results[0]['support_floor20']['active_modes']}")
    print(f"elapsed seconds: {elapsed:.1f}")
    print("best by kind:")
    for kind in sorted(by_kind):
        row = by_kind[kind]["best"]
        print(f"  {kind:13s} count={by_kind[kind]['count']:5d} best={row['maximized_value']:.15f} gap={row['gap_to_target']:+.3e}")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "mu": args.mu,
        "target": C3_TARGET,
        "start_mix": mix,
        "task_seeds": args.task_seeds,
        "seed": args.seed,
        "workers": workers,
        "problem_modes": int(problem["N"]),
        "problem_triads": int(len(problem["ell_idx"])),
        "active_indices": [int(index) for index in active],
        "orbit_supports": [[int(index) for index in support] for support in orbit_supports],
        "inactive_modes": int(np.sum(inactive_mask)),
        "options": options | {"active_indices": [int(index) for index in active]},
        "elapsed_seconds": elapsed,
        "best": results[0],
        "best_by_kind": by_kind,
        "top_results": results[: args.top],
        "all_results": results,
    }
    output = Path(args.output) if args.output else Path("results") / f"k3_fullblock_parallel_census_{args.mode}_{datetime.now():%Y%m%d_%H%M%S}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()