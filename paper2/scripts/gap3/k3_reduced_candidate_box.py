#!/usr/bin/env python3
"""Optimize the boundary of a reduced-kernel candidate neighborhood."""

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

from scripts.gap3.k3_three_variable_reduction import k3_reduced  # noqa: E402


CANDIDATE = np.array([-0.22384360783153906, -1.3921663090685699, 3.6136991174663304], dtype=float)


def parse_floats(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def reduced_value(vector: np.ndarray) -> float:
    return k3_reduced(math.exp(float(vector[0])), math.exp(float(vector[1])), float(vector[2]))


def optimize_face(
    bounds: list[tuple[float, float]],
    fixed_index: int,
    side: int,
    seed: int,
    maxiter: int,
    popsize: int,
    local_maxiter: int,
) -> dict:
    fixed_value = bounds[fixed_index][side]
    free_indices = [index for index in range(3) if index != fixed_index]
    free_bounds = [bounds[index] for index in free_indices]

    def lift(free_vector: np.ndarray) -> np.ndarray:
        vector = np.zeros(3, dtype=float)
        vector[fixed_index] = fixed_value
        for index, value in zip(free_indices, free_vector):
            vector[index] = value
        return vector

    result = differential_evolution(
        lambda free_vector: -reduced_value(lift(free_vector)),
        free_bounds,
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-12,
        polish=False,
        updating="immediate",
        workers=1,
    )
    polished = minimize(
        lambda free_vector: -reduced_value(lift(free_vector)),
        np.array(result.x, dtype=float),
        method="L-BFGS-B",
        bounds=free_bounds,
        options={"ftol": 1e-16, "gtol": 1e-12, "maxiter": local_maxiter},
    )
    vector = lift(np.array(polished.x, dtype=float))
    return {
        "fixed_index": fixed_index,
        "fixed_variable": ["log_t", "log_r", "theta"][fixed_index],
        "side": "lo" if side == 0 else "hi",
        "fixed_value": fixed_value,
        "value": reduced_value(vector),
        "vector": [float(component) for component in vector],
        "success": bool(polished.success),
        "message": str(polished.message),
        "de_value": -float(result.fun),
        "de_vector": [float(component) for component in result.x],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius", default="0.15,0.15,0.15")
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--maxiter", type=int, default=160)
    parser.add_argument("--popsize", type=int, default=20)
    parser.add_argument("--local-maxiter", type=int, default=3000)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    radius = np.array(parse_floats(args.radius), dtype=float)
    if radius.shape != (3,) or np.any(radius <= 0):
        raise ValueError("--radius must contain three positive comma-separated floats")
    bounds = [(float(CANDIDATE[index] - radius[index]), float(CANDIDATE[index] + radius[index])) for index in range(3)]
    center_value = reduced_value(CANDIDATE)

    print("k=3 reduced candidate box")
    print("==========================")
    print(f"center value: {center_value:.17g}")
    print(f"bounds: {bounds}")
    start_time = time.time()
    faces = []
    for fixed_index in range(3):
        for side in range(2):
            face = optimize_face(
                bounds,
                fixed_index,
                side,
                args.seed + 2 * fixed_index + side,
                args.maxiter,
                args.popsize,
                args.local_maxiter,
            )
            face["drop_from_center"] = center_value - face["value"]
            faces.append(face)
            print(
                f"{face['fixed_variable']:5s} {face['side']:2s}: "
                f"value={face['value']:.17g} drop={face['drop_from_center']:.3e}",
                flush=True,
            )
    best_face = max(faces, key=lambda item: item["value"])
    elapsed = time.time() - start_time
    print("\nSummary")
    print("-------")
    print(f"best boundary value: {best_face['value']:.17g}")
    print(f"minimum boundary drop: {center_value - best_face['value']:.6e}")
    print(f"elapsed seconds: {elapsed:.1f}")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "variables": ["log_t", "log_r", "theta"],
        "candidate": [float(component) for component in CANDIDATE],
        "radius": [float(component) for component in radius],
        "bounds": [[lo, hi] for lo, hi in bounds],
        "center_value": center_value,
        "faces": faces,
        "best_boundary_face": best_face,
        "minimum_boundary_drop": center_value - best_face["value"],
        "elapsed_seconds": elapsed,
    }
    if not args.no_save:
        output = Path(args.output) if args.output else Path("results") / f"k3_reduced_candidate_box_{datetime.now():%Y%m%d_%H%M%S}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved: {output}")


if __name__ == "__main__":
    main()