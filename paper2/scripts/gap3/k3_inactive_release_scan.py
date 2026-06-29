#!/usr/bin/env python3
"""Search inactive k=3 release directions for the 9-mode candidate.

This is a numerical KKT diagnostic for the closed-form k=3 candidate.  For
each inactive mode in the requested k=3 scope, it minimizes the one-sided
release coefficient

    d(-R)/d(exp(loga))

over that mode's three angular/polarization variables.  A positive minimum
means that releasing that inactive amplitude decreases R to first order.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.gap3.k3_active_set_verify import (  # noqa: E402
    build_problem_scope,
    embed_support,
    ratio_and_grad,
)
from scripts.gap3.k3_active_hessian_check import refine_active_support  # noqa: E402
from scripts.gap3.k3_closed_form_probe import (  # noqa: E402
    build_prob,
    support_from_warm,
)


ANGLE_BOUNDS = ((0.0, math.pi / 2), (0.0, 2 * math.pi), (0.0, 2 * math.pi))


def parse_float_list(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def release_coefficient(
    problem: dict,
    base_params: np.ndarray,
    mode_index: int,
    angles: np.ndarray,
    release_floor: float,
) -> float:
    params = base_params.copy()
    offset = 4 * mode_index
    params[offset : offset + 3] = angles
    params[offset + 3] = release_floor
    _, grad = ratio_and_grad(problem, params)
    return float(grad[offset + 3] / math.exp(release_floor))


def random_angles(rng: np.random.Generator, count: int) -> np.ndarray:
    lo = np.array([bound[0] for bound in ANGLE_BOUNDS])
    hi = np.array([bound[1] for bound in ANGLE_BOUNDS])
    return rng.uniform(lo, hi, size=(count, 3))


def scan_mode(
    problem: dict,
    base_params: np.ndarray,
    mode_index: int,
    release_floor: float,
    check_floors: list[float],
    samples: int,
    polish_starts: int,
    maxiter: int,
    rng: np.random.Generator,
) -> dict:
    sample_angles = random_angles(rng, samples)
    sampled = [
        (release_coefficient(problem, base_params, mode_index, angles, release_floor), angles)
        for angles in sample_angles
    ]
    sampled.sort(key=lambda item: item[0])

    best_value = sampled[0][0]
    best_angles = sampled[0][1].copy()
    local_results = []

    def objective(angles: np.ndarray) -> float:
        return release_coefficient(problem, base_params, mode_index, angles, release_floor)

    for value, start in sampled[:polish_starts]:
        result = minimize(
            objective,
            start,
            method="Nelder-Mead",
            options={"maxiter": maxiter, "xatol": 1e-12, "fatol": 1e-14},
        )
        angles = np.array(result.x, dtype=float)
        angles[0] = min(max(angles[0], ANGLE_BOUNDS[0][0]), ANGLE_BOUNDS[0][1])
        angles[1:] = np.mod(angles[1:], 2 * math.pi)
        polished_value = objective(angles)
        local_results.append(
            {
                "start_value": value,
                "polished_value": polished_value,
                "success": bool(result.success),
                "iterations": int(result.nit),
                "function_evaluations": int(result.nfev),
            }
        )
        if polished_value < best_value:
            best_value = polished_value
            best_angles = angles.copy()

    floor_checks = {
        str(floor): release_coefficient(problem, base_params, mode_index, best_angles, floor)
        for floor in check_floors
    }
    wavevector = tuple(int(component) for component in problem["wavevecs"][mode_index])
    return {
        "mode_index": mode_index,
        "shell": int(problem["k2s"][mode_index]),
        "wavevector": wavevector,
        "sample_min": sampled[0][0],
        "sample_max": sampled[-1][0],
        "best_value": best_value,
        "best_angles": [float(value) for value in best_angles],
        "check_floors": floor_checks,
        "local_results": local_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-floor", type=float, default=-20.0)
    parser.add_argument("--inactive-floor", type=float, default=-80.0)
    parser.add_argument("--check-floors", default="-16,-20,-24")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--polish-starts", type=int, default=4)
    parser.add_argument("--maxiter", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--active-refine-starts", type=int, default=0)
    parser.add_argument("--active-refine-scales", default="1e-4,1e-3,1e-2,5e-2")
    parser.add_argument("--active-refine-seed", type=int, default=20260530)
    parser.add_argument("--scope", choices=("nucleus", "full-block"), default="nucleus")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print("k=3 inactive release scan")
    print("==========================")
    support_indices, support_modes, support_start = support_from_warm()
    support_problem = build_prob(support_modes)
    support_value, support_params, support_grad, refine_rows = refine_active_support(
        support_problem,
        support_start,
        args.active_refine_starts,
        parse_float_list(args.active_refine_scales),
        args.active_refine_seed,
    )
    full_problem = build_problem_scope(args.scope)
    base_params, active_indices = embed_support(full_problem, support_modes, support_params, args.inactive_floor)
    active_set = set(active_indices)
    inactive_indices = [index for index in range(full_problem["N"]) if index not in active_set]

    print(f"support value:    {support_value:.17g}")
    print(f"support grad max: {support_grad:.3e}")
    print(f"active refine tries: {len(refine_rows)}")
    print(f"scope:            {args.scope}")
    print(f"problem:          N={full_problem['N']} T={len(full_problem['ell_idx'])}")
    print(f"inactive modes:   {len(inactive_indices)}")
    print(f"release floor:    {args.release_floor:g}")
    print(f"other inactive floor: {args.inactive_floor:g}")

    rng = np.random.default_rng(args.seed)
    start_time = time.time()
    check_floors = parse_float_list(args.check_floors)
    rows = []
    for ordinal, mode_index in enumerate(inactive_indices, 1):
        row = scan_mode(
            full_problem,
            base_params,
            mode_index,
            args.release_floor,
            check_floors,
            args.samples,
            args.polish_starts,
            args.maxiter,
            rng,
        )
        rows.append(row)
        print(
            f"{ordinal:02d}/{len(inactive_indices)} idx={mode_index:2d} "
            f"shell={row['shell']:2d} mode={row['wavevector']} "
            f"min={row['best_value']:+.6e} sample={row['sample_min']:+.6e}",
            flush=True,
        )

    rows_sorted = sorted(rows, key=lambda item: item["best_value"])
    minimum = rows_sorted[0]
    negative = [row for row in rows_sorted if row["best_value"] < -1e-10]
    elapsed = time.time() - start_time

    print("\nSummary")
    print("-------")
    print(
        f"minimum release coefficient: {minimum['best_value']:+.12e} "
        f"at idx={minimum['mode_index']} mode={minimum['wavevector']}"
    )
    print(f"negative coefficients below -1e-10: {len(negative)}")
    print(f"elapsed seconds: {elapsed:.1f}")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "support_indices": support_indices,
        "support_modes": support_modes,
        "support_value_float64": support_value,
        "support_grad_max": support_grad,
        "active_refine_starts_per_scale": args.active_refine_starts,
        "active_refine_scales": parse_float_list(args.active_refine_scales),
        "active_refine_top": sorted(refine_rows, key=lambda item: item["value"], reverse=True)[:20],
        "scope": args.scope,
        "problem_modes": int(full_problem["N"]),
        "problem_triads": int(len(full_problem["ell_idx"])),
        "release_floor": args.release_floor,
        "inactive_floor": args.inactive_floor,
        "check_floors": check_floors,
        "samples": args.samples,
        "polish_starts": args.polish_starts,
        "seed": args.seed,
        "elapsed_seconds": elapsed,
        "minimum": minimum,
        "negative_count_below_1e-10": len(negative),
        "inactive_modes": rows_sorted,
    }

    if not args.no_save:
        output = Path(args.output) if args.output else Path("results") / f"k3_inactive_release_scan_{datetime.now():%Y%m%d_%H%M%S}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved: {output}")


if __name__ == "__main__":
    main()