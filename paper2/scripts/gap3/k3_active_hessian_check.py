#!/usr/bin/env python3
"""Finite-difference Hessian spectrum for the k=3 nine-mode active support.

This is a targeted numerical local-maximality diagnostic for the algebraic
k=3 candidate.  It optimizes the nine active modes, differentiates the
analytic gradient of -R by centered differences, and records the Hessian
spectrum.  For a local maximum of R, the Hessian of -R should be positive
semidefinite up to symmetry/scale flat directions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.gap3.k3_closed_form_probe import (  # noqa: E402
    build_prob,
    mpmath_value,
    optimise_current_objective,
    support_from_warm,
)
from scripts.gap3.max_b_over_keff import neg_ratio_and_grad  # noqa: E402


def parse_float_list(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def objective_gradient(problem: dict, params: np.ndarray) -> np.ndarray:
    _, gradient = neg_ratio_and_grad(
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
    return np.asarray(gradient, dtype=float)


def finite_difference_hessian(problem: dict, params: np.ndarray, step: float) -> np.ndarray:
    dimension = len(params)
    hessian = np.zeros((dimension, dimension), dtype=float)
    for column in range(dimension):
        delta = np.zeros(dimension, dtype=float)
        delta[column] = step
        grad_plus = objective_gradient(problem, params + delta)
        grad_minus = objective_gradient(problem, params - delta)
        hessian[:, column] = (grad_plus - grad_minus) / (2.0 * step)
    return 0.5 * (hessian + hessian.T)


def refine_active_support(
    problem: dict,
    start: np.ndarray,
    starts_per_scale: int,
    scales: list[float],
    seed: int,
) -> tuple[float, np.ndarray, float, list[dict]]:
    best_value, best_params, best_grad = optimise_current_objective(problem, start)
    rows = [
        {
            "label": "warm_start",
            "value": best_value,
            "gradient_max_abs": best_grad,
        }
    ]
    if starts_per_scale <= 0:
        return best_value, best_params, best_grad, rows

    rng = np.random.default_rng(seed)
    lower = np.array([0.0, 0.0, 0.0, -8.0] * problem["N"], dtype=float)
    upper = np.array([np.pi / 2, 2 * np.pi, 2 * np.pi, 8.0] * problem["N"], dtype=float)
    for scale in scales:
        for index in range(starts_per_scale):
            trial = start.copy()
            for mode_index in range(problem["N"]):
                trial[4 * mode_index : 4 * mode_index + 4] += rng.normal(0.0, scale, size=4)
            trial = np.minimum(np.maximum(trial, lower), upper)
            value, params, grad = optimise_current_objective(problem, trial)
            rows.append(
                {
                    "label": f"scale={scale:g}:{index}",
                    "value": value,
                    "gradient_max_abs": grad,
                }
            )
            if value > best_value:
                best_value = value
                best_params = params
                best_grad = grad
    return best_value, best_params, best_grad, rows


def classify(eigenvalues: np.ndarray, tolerance: float) -> dict:
    return {
        "positive": int(np.sum(eigenvalues > tolerance)),
        "flat": int(np.sum(np.abs(eigenvalues) <= tolerance)),
        "negative": int(np.sum(eigenvalues < -tolerance)),
        "min": float(eigenvalues[0]),
        "max": float(eigenvalues[-1]),
        "smallest_12": [float(value) for value in eigenvalues[:12]],
        "largest_12": [float(value) for value in eigenvalues[-12:]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", default="1e-4,3e-5,1e-5")
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--mp-dps", type=int, default=90)
    parser.add_argument("--refine-starts", type=int, default=0)
    parser.add_argument("--refine-scales", default="1e-4,1e-3,1e-2,5e-2")
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print("k=3 active-support Hessian check")
    print("=================================")
    start_time = time.time()
    support_indices, support_modes, start = support_from_warm()
    problem = build_prob(support_modes)
    value, params, grad_max, refine_rows = refine_active_support(
        problem,
        start,
        args.refine_starts,
        parse_float_list(args.refine_scales),
        args.seed,
    )
    value_mp = mpmath_value(problem, params, dps=args.mp_dps)
    gradient = objective_gradient(problem, params)

    print(f"active modes: {len(support_modes)}")
    print(f"parameters:   {len(params)}")
    print(f"triads:       {len(problem['ell_idx'])}")
    print(f"value f64:    {value:.17g}")
    print(f"value mp:     {value_mp}")
    print(f"grad max:     {float(np.max(np.abs(gradient))):.3e}")
    print(f"refine tries: {len(refine_rows)}")

    rows = []
    for step in parse_float_list(args.steps):
        hessian = finite_difference_hessian(problem, params, step)
        eigenvalues = np.linalg.eigvalsh(hessian)
        row = {
            "step": step,
            "hessian_of": "negative_objective_-R",
            "tolerance": args.tolerance,
            "classification": classify(eigenvalues, args.tolerance),
        }
        rows.append(row)
        cls = row["classification"]
        print(
            f"step={step:.1e}: pos={cls['positive']} flat={cls['flat']} "
            f"neg={cls['negative']} min={cls['min']:+.3e} max={cls['max']:+.3e}",
            flush=True,
        )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "support_indices": support_indices,
        "support_modes": support_modes,
        "active_mode_count": len(support_modes),
        "parameter_count": len(params),
        "triads": int(len(problem["ell_idx"])),
        "value_float64": value,
        "value_mpmath": str(value_mp),
        "gradient_max_abs": float(np.max(np.abs(gradient))),
        "refine_starts_per_scale": args.refine_starts,
        "refine_scales": parse_float_list(args.refine_scales),
        "refine_results": sorted(refine_rows, key=lambda item: item["value"], reverse=True)[:20],
        "checks": rows,
        "elapsed_seconds": time.time() - start_time,
    }
    if not args.no_save:
        output = Path(args.output) if args.output else Path("results") / f"k3_active_hessian_check_{datetime.now():%Y%m%d_%H%M%S}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved: {output}")


if __name__ == "__main__":
    main()