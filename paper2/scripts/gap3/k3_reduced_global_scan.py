#!/usr/bin/env python3
"""Global-window and boundary diagnostic for the reduced k=3 kernel.

This searches the three-variable reduced objective in a broad compact window

    (log t, log r, theta) in [a,b] x [c,d] x [0,2*pi]

and separately optimizes the four log-coordinate boundary faces.  It is a
numerical diagnostic, not an interval proof.  Its job is to catch competing
reduced-kernel peaks and to quantify the boundary margin before a future
real-root/interval certificate is attempted.
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
from scipy.optimize import differential_evolution, minimize

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.gap3.k3_three_variable_reduction import k3_reduced, recover_pqh  # noqa: E402


def parse_bounds(text: str) -> tuple[float, float]:
    left, right = [float(part.strip()) for part in text.split(",", 1)]
    if not left < right:
        raise ValueError(f"invalid bounds {text!r}")
    return left, right


def safe_reduced(vector: np.ndarray) -> float:
    log_t, log_r, theta = vector
    try:
        value = k3_reduced(math.exp(float(log_t)), math.exp(float(log_r)), float(theta))
    except (ValueError, OverflowError, ZeroDivisionError):
        return -math.inf
    if not math.isfinite(value):
        return -math.inf
    return float(value)


def minimize_negative(start: np.ndarray, bounds: list[tuple[float, float]], maxiter: int) -> dict:
    result = minimize(
        lambda vector: -safe_reduced(vector),
        start,
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-16, "gtol": 1e-12, "maxiter": maxiter},
    )
    vector = np.array(result.x, dtype=float)
    vector[2] = vector[2] % (2 * math.pi)
    value = safe_reduced(vector)
    return {
        "value": value,
        "vector": [float(component) for component in vector],
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
    }


def random_scan(
    bounds: list[tuple[float, float]],
    samples: int,
    top_starts: int,
    seed: int,
    maxiter: int,
) -> dict:
    rng = np.random.default_rng(seed)
    lo = np.array([bound[0] for bound in bounds], dtype=float)
    hi = np.array([bound[1] for bound in bounds], dtype=float)
    points = rng.uniform(lo, hi, size=(samples, 3))
    values = np.array([safe_reduced(point) for point in points], dtype=float)
    order = np.argsort(values)[::-1]
    polished = [minimize_negative(points[index], bounds, maxiter) for index in order[:top_starts]]
    polished.sort(key=lambda item: item["value"], reverse=True)
    finite_values = values[np.isfinite(values)]
    return {
        "samples": samples,
        "finite_samples": int(finite_values.size),
        "sample_best_value": float(finite_values.max()) if finite_values.size else None,
        "sample_quantiles": {
            "q50": float(np.quantile(finite_values, 0.50)) if finite_values.size else None,
            "q90": float(np.quantile(finite_values, 0.90)) if finite_values.size else None,
            "q99": float(np.quantile(finite_values, 0.99)) if finite_values.size else None,
            "q999": float(np.quantile(finite_values, 0.999)) if finite_values.size else None,
        },
        "polished_top": polished[:10],
    }


def differential_scan(
    bounds: list[tuple[float, float]],
    seed: int,
    maxiter: int,
    popsize: int,
    local_maxiter: int,
) -> dict:
    result = differential_evolution(
        lambda vector: -safe_reduced(vector),
        bounds,
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-11,
        polish=False,
        updating="immediate",
        workers=1,
    )
    polished = minimize_negative(np.array(result.x, dtype=float), bounds, local_maxiter)
    return {
        "de_value": -float(result.fun),
        "de_vector": [float(component) for component in result.x],
        "de_success": bool(result.success),
        "de_message": str(result.message),
        "de_iterations": int(result.nit),
        "de_function_evaluations": int(result.nfev),
        "polished": polished,
    }


def face_scan(
    bounds: list[tuple[float, float]],
    fixed_index: int,
    fixed_value: float,
    seed: int,
    maxiter: int,
    popsize: int,
    local_maxiter: int,
) -> dict:
    free_indices = [index for index in range(3) if index != fixed_index]
    free_bounds = [bounds[index] for index in free_indices]

    def lift(free_vector: np.ndarray) -> np.ndarray:
        vector = np.zeros(3, dtype=float)
        vector[fixed_index] = fixed_value
        for free_index, value in zip(free_indices, free_vector):
            vector[free_index] = value
        return vector

    result = differential_evolution(
        lambda free_vector: -safe_reduced(lift(free_vector)),
        free_bounds,
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-11,
        polish=False,
        updating="immediate",
        workers=1,
    )
    lifted = lift(np.array(result.x, dtype=float))
    polished_free = minimize(
        lambda free_vector: -safe_reduced(lift(free_vector)),
        np.array(result.x, dtype=float),
        method="L-BFGS-B",
        bounds=free_bounds,
        options={"ftol": 1e-16, "gtol": 1e-12, "maxiter": local_maxiter},
    )
    polished_vector = lift(np.array(polished_free.x, dtype=float))
    polished_vector[2] = polished_vector[2] % (2 * math.pi)
    return {
        "fixed_index": fixed_index,
        "fixed_value": fixed_value,
        "free_indices": free_indices,
        "de_value": safe_reduced(lift(np.array(result.x, dtype=float))),
        "de_vector": [float(component) for component in lifted],
        "de_success": bool(result.success),
        "de_message": str(result.message),
        "polished_value": safe_reduced(polished_vector),
        "polished_vector": [float(component) for component in polished_vector],
        "polished_success": bool(polished_free.success),
        "polished_message": str(polished_free.message),
    }


def vector_details(vector: list[float]) -> dict:
    log_t, log_r, theta = vector
    t_ratio = math.exp(log_t)
    radius = math.exp(log_r)
    p_value, q_value, h_value = recover_pqh(t_ratio, radius, theta % (2 * math.pi))
    return {
        "t": t_ratio,
        "r": radius,
        "theta": theta % (2 * math.pi),
        "p": p_value,
        "q": q_value,
        "h": h_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-t-bounds", default="-10,6")
    parser.add_argument("--log-r-bounds", default="-14,6")
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--top-starts", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--de-maxiter", type=int, default=220)
    parser.add_argument("--face-maxiter", type=int, default=180)
    parser.add_argument("--popsize", type=int, default=20)
    parser.add_argument("--local-maxiter", type=int, default=3000)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    bounds = [
        parse_bounds(args.log_t_bounds),
        parse_bounds(args.log_r_bounds),
        (0.0, 2 * math.pi),
    ]
    start_time = time.time()
    print("k=3 reduced global scan")
    print("========================")
    print(f"bounds: log_t={bounds[0]} log_r={bounds[1]} theta={bounds[2]}")

    random_result = random_scan(bounds, args.samples, args.top_starts, args.seed, args.local_maxiter)
    print(f"random sample best: {random_result['sample_best_value']:.17g}")
    if random_result["polished_top"]:
        print(f"random polished best: {random_result['polished_top'][0]['value']:.17g}")

    differential_result = differential_scan(
        bounds,
        args.seed + 101,
        args.de_maxiter,
        args.popsize,
        args.local_maxiter,
    )
    print(f"DE polished best: {differential_result['polished']['value']:.17g}")

    face_specs = [
        ("log_t_min", 0, bounds[0][0]),
        ("log_t_max", 0, bounds[0][1]),
        ("log_r_min", 1, bounds[1][0]),
        ("log_r_max", 1, bounds[1][1]),
    ]
    faces = {}
    for ordinal, (label, fixed_index, fixed_value) in enumerate(face_specs, 1):
        result = face_scan(
            bounds,
            fixed_index,
            fixed_value,
            args.seed + 1000 + ordinal,
            args.face_maxiter,
            args.popsize,
            args.local_maxiter,
        )
        faces[label] = result
        print(f"face {label:9s}: {result['polished_value']:.17g}")

    candidates = [differential_result["polished"]] + random_result["polished_top"]
    best = max(candidates, key=lambda item: item["value"])
    best_with_details = dict(best)
    best_with_details["details"] = vector_details(best["vector"])
    best_face_label, best_face = max(faces.items(), key=lambda item: item[1]["polished_value"])
    elapsed = time.time() - start_time

    print("\nSummary")
    print("-------")
    print(f"best interior value: {best['value']:.17g}")
    print(f"best interior details: {best_with_details['details']}")
    print(f"best boundary face: {best_face_label} value={best_face['polished_value']:.17g}")
    print(f"interior minus best face: {best['value'] - best_face['polished_value']:.6e}")
    print(f"elapsed seconds: {elapsed:.1f}")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "bounds": bounds,
        "random_scan": random_result,
        "differential_evolution": differential_result,
        "boundary_faces": faces,
        "best_interior": best_with_details,
        "best_boundary_face_label": best_face_label,
        "best_boundary_face": best_face,
        "interior_minus_best_boundary_face": best["value"] - best_face["polished_value"],
        "elapsed_seconds": elapsed,
    }

    if not args.no_save:
        output = Path(args.output) if args.output else Path("results") / f"k3_reduced_global_scan_{datetime.now():%Y%m%d_%H%M%S}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved: {output}")


if __name__ == "__main__":
    main()