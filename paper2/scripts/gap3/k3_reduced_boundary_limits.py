#!/usr/bin/env python3
"""Boundary-limit diagnostics for the reduced k=3 kernel.

The reduced objective has two positive shape variables, t and r.  This script
optimizes the explicit limiting objectives on the boundary faces r=0,
t=infinity, and r=infinity.  It is numerical evidence for boundary exclusion,
not an interval certificate.
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

from scripts.gap3.k3_three_variable_reduction import k3_reduced, q_theta  # noqa: E402


def parse_bounds(text: str) -> tuple[float, float]:
    left, right = [float(part.strip()) for part in text.split(",", 1)]
    if not left < right:
        raise ValueError(f"invalid bounds {text!r}")
    return left, right


def direction_from_s_u(t_ratio: float, s_gain: float, u_gain: float) -> float:
    if abs(u_gain) < 1e-15:
        return t_ratio * s_gain
    root = math.sqrt(s_gain * s_gain + 8 * u_gain * u_gain * t_ratio * t_ratio)
    sin_phi = (root - s_gain) / (4 * u_gain * t_ratio)
    sin_phi = max(0.0, min(1.0, sin_phi))
    cos_phi = math.sqrt(max(0.0, 1.0 - sin_phi * sin_phi))
    return t_ratio * cos_phi * (s_gain + u_gain * t_ratio * sin_phi)


def scale_factor(alpha: float, beta: float, t_ratio: float) -> float:
    a_t = 5 + 7 * t_ratio * t_ratio
    g_t = 25 + 49 * t_ratio * t_ratio
    ell = math.sqrt(1 + 8 * a_t * beta / (alpha * g_t))
    return math.sqrt(8 / (alpha * a_t * g_t)) * math.sqrt(1 + ell) / ((3 + ell) ** 1.5)


def r_zero_value(log_t: float) -> float:
    t_ratio = math.exp(log_t)
    s_gain = math.sqrt(1259 - 108 * math.sqrt(35)) + math.sqrt(1259 + 108 * math.sqrt(35))
    return math.sqrt(70) * direction_from_s_u(t_ratio, s_gain, 0.0) * scale_factor(2.0, 8.0, t_ratio) / 280


def t_infty_value(log_r: float, theta: float) -> float:
    radius = math.exp(log_r)
    alpha = 2 + 5 * radius * radius
    beta = 8 + 25 * radius * radius
    ell = math.sqrt(1 + (8 * beta) / (7 * alpha))
    u_gain = radius * math.sqrt(max(0.0, 2 * q_theta(theta)))
    scale_limit = math.sqrt(8 / (343 * alpha)) * math.sqrt(1 + ell) / ((3 + ell) ** 1.5)
    return math.sqrt(70) * (u_gain / 2) * scale_limit / 280


def r_infty_value(log_t: float, theta: float) -> float:
    t_ratio = math.exp(log_t)
    s_gain = 2 * math.sqrt(637)
    u_gain = math.sqrt(max(0.0, 2 * q_theta(theta)))
    a_t = 5 + 7 * t_ratio * t_ratio
    g_t = 25 + 49 * t_ratio * t_ratio
    ell = math.sqrt(1 + 40 * a_t / g_t)
    scale_limit = math.sqrt(8 / (5 * a_t * g_t)) * math.sqrt(1 + ell) / ((3 + ell) ** 1.5)
    return math.sqrt(70) * direction_from_s_u(t_ratio, s_gain, u_gain) * scale_limit / 280


def optimize_face(label: str, func, bounds: list[tuple[float, float]], seed: int, maxiter: int, popsize: int) -> dict:
    result = differential_evolution(
        lambda vector: -func(*vector),
        bounds,
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-12,
        polish=False,
        updating="immediate",
        workers=1,
    )
    polished = minimize(
        lambda vector: -func(*vector),
        np.array(result.x, dtype=float),
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-16, "gtol": 1e-13, "maxiter": 5000},
    )
    return {
        "label": label,
        "value": float(func(*polished.x)),
        "variables": [float(component) for component in polished.x],
        "de_value": -float(result.fun),
        "de_variables": [float(component) for component in result.x],
        "success": bool(polished.success),
        "message": str(polished.message),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-t-bounds", default="-20,12")
    parser.add_argument("--log-r-bounds", default="-30,12")
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--maxiter", type=int, default=320)
    parser.add_argument("--popsize", type=int, default=24)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    log_t_bounds = parse_bounds(args.log_t_bounds)
    log_r_bounds = parse_bounds(args.log_r_bounds)
    theta_bounds = (0.0, 2 * math.pi)
    interior_value = k3_reduced(0.79944015077205111, 0.24853631456081229, 3.6136991174663304)

    print("k=3 reduced boundary limits")
    print("============================")
    start_time = time.time()
    faces = [
        optimize_face("r_zero", lambda log_t: r_zero_value(log_t), [log_t_bounds], args.seed + 1, args.maxiter, args.popsize),
        optimize_face(
            "t_infinity",
            lambda log_r, theta: t_infty_value(log_r, theta),
            [log_r_bounds, theta_bounds],
            args.seed + 2,
            args.maxiter,
            args.popsize,
        ),
        optimize_face(
            "r_infinity",
            lambda log_t, theta: r_infty_value(log_t, theta),
            [log_t_bounds, theta_bounds],
            args.seed + 3,
            args.maxiter,
            args.popsize,
        ),
    ]
    for face in faces:
        print(f"{face['label']:11s}: value={face['value']:.17g} variables={face['variables']}")

    best_face = max(faces, key=lambda item: item["value"])
    elapsed = time.time() - start_time
    print("\nSummary")
    print("-------")
    print(f"interior candidate: {interior_value:.17g}")
    print(f"best boundary limit: {best_face['label']} value={best_face['value']:.17g}")
    print(f"interior minus boundary: {interior_value - best_face['value']:.6e}")
    print(f"elapsed seconds: {elapsed:.1f}")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "interior_candidate_value": interior_value,
        "log_t_bounds": log_t_bounds,
        "log_r_bounds": log_r_bounds,
        "boundary_limits": faces,
        "best_boundary_limit": best_face,
        "interior_minus_best_boundary_limit": interior_value - best_face["value"],
        "elapsed_seconds": elapsed,
    }

    if not args.no_save:
        output = Path(args.output) if args.output else Path("results") / f"k3_reduced_boundary_limits_{datetime.now():%Y%m%d_%H%M%S}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved: {output}")


if __name__ == "__main__":
    main()