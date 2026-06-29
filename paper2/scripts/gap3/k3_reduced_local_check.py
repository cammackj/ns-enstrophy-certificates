#!/usr/bin/env python3
"""Local maximality diagnostic for the reduced k=3 kernel.

This checks the three-variable objective obtained after analytically eliminating
p, q, and h from the k=3 equivariant kernel.  It is numerical evidence, not a
root-isolation proof: the purpose is to record the stationary residual and the
finite-difference Hessian spectrum at the closed-form candidate.
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

import mpmath as mp
import numpy as np
from scipy.optimize import minimize

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.gap3.k3_three_variable_reduction import k3_reduced, recover_pqh


INITIAL = {
    "p": 0.6588123322667734,
    "q": 0.5254332985113650,
    "r": 0.2485363066732576,
    "h": 0.0362319980897601,
    "theta": 3.613699113102433,
}


def reduced_float(vector: np.ndarray) -> float:
    log_t, log_r, theta = vector
    return k3_reduced(math.exp(log_t), math.exp(log_r), theta)


def p_sigma_mp(sigma: int, radius: mp.mpf, theta: mp.mpf) -> mp.mpf:
    return (
        mp.mpf(1259)
        - 108 * sigma * mp.sqrt(35)
        + 637 * radius * radius
        + 2
        * radius
        * (
            (162 * sigma * mp.sqrt(7) - 299 * mp.sqrt(5)) * mp.cos(theta)
            - (180 * mp.sqrt(2) + 39 * sigma * mp.sqrt(70)) * mp.sin(theta)
        )
    )


def q_theta_mp(theta: mp.mpf) -> mp.mpf:
    return 575 + 325 * mp.cos(2 * theta) - 150 * mp.sqrt(10) * mp.sin(2 * theta)


def direction_gain_mp(t_ratio: mp.mpf, radius: mp.mpf, theta: mp.mpf) -> mp.mpf:
    s_gain = mp.sqrt(p_sigma_mp(1, radius, theta)) + mp.sqrt(p_sigma_mp(-1, radius, theta))
    u_gain = radius * mp.sqrt(2 * q_theta_mp(theta))
    root = mp.sqrt(s_gain * s_gain + 8 * u_gain * u_gain * t_ratio * t_ratio)
    sin_phi = (root - s_gain) / (4 * u_gain * t_ratio)
    cos_phi = mp.sqrt(1 - sin_phi * sin_phi)
    return t_ratio * cos_phi * (s_gain + u_gain * t_ratio * sin_phi)


def scale_gain_mp(t_ratio: mp.mpf, radius: mp.mpf) -> mp.mpf:
    alpha = 2 + 5 * radius * radius
    beta = 8 + 25 * radius * radius
    a_t = 5 + 7 * t_ratio * t_ratio
    g_t = 25 + 49 * t_ratio * t_ratio
    ell = mp.sqrt(1 + 8 * a_t * beta / (alpha * g_t))
    return mp.sqrt(8 / (alpha * a_t * g_t)) * mp.sqrt(1 + ell) / ((3 + ell) ** mp.mpf("1.5"))


def reduced_mp(vector: list[mp.mpf]) -> mp.mpf:
    log_t, log_r, theta = vector
    t_ratio = mp.exp(log_t)
    radius = mp.exp(log_r)
    return mp.sqrt(70) * direction_gain_mp(t_ratio, radius, theta) * scale_gain_mp(t_ratio, radius) / 280


def optimize_reduced(maxiter: int) -> np.ndarray:
    t0 = math.hypot(INITIAL["q"], INITIAL["h"]) / INITIAL["p"]
    x0 = np.array([math.log(t0), math.log(INITIAL["r"]), INITIAL["theta"]], dtype=float)

    result = minimize(
        lambda vector: -reduced_float(vector),
        x0,
        method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": 1e-14, "fatol": 1e-16},
    )
    return np.array(result.x, dtype=float)


def finite_difference_derivatives(vector: np.ndarray, step: mp.mpf, dps: int) -> tuple[list[mp.mpf], list[list[mp.mpf]]]:
    mp.mp.dps = dps
    point = [mp.mpf(str(value)) for value in vector]
    f0 = reduced_mp(point)
    gradient: list[mp.mpf] = []
    hessian = [[mp.mpf(0) for _ in range(3)] for _ in range(3)]

    for i in range(3):
        plus = point.copy()
        minus = point.copy()
        plus[i] += step
        minus[i] -= step
        f_plus = reduced_mp(plus)
        f_minus = reduced_mp(minus)
        gradient.append((f_plus - f_minus) / (2 * step))
        hessian[i][i] = (f_plus - 2 * f0 + f_minus) / (step * step)

    for i in range(3):
        for j in range(i + 1, 3):
            pp = point.copy()
            pm = point.copy()
            mpv = point.copy()
            mm = point.copy()
            pp[i] += step
            pp[j] += step
            pm[i] += step
            pm[j] -= step
            mpv[i] -= step
            mpv[j] += step
            mm[i] -= step
            mm[j] -= step
            value = (reduced_mp(pp) - reduced_mp(pm) - reduced_mp(mpv) + reduced_mp(mm)) / (4 * step * step)
            hessian[i][j] = value
            hessian[j][i] = value

    return gradient, hessian


def derivative_rows(vector: np.ndarray, steps: list[float], dps: int) -> list[dict]:
    rows = []
    for step_float in steps:
        gradient, hessian = finite_difference_derivatives(vector, mp.mpf(str(step_float)), dps)
        hessian_np = np.array([[float(value) for value in row] for row in hessian], dtype=float)
        eigenvalues = np.linalg.eigvalsh(hessian_np)
        rows.append(
            {
                "step": step_float,
                "gradient": [str(value) for value in gradient],
                "gradient_inf": float(max(abs(value) for value in gradient)),
                "hessian": [[str(value) for value in row] for row in hessian],
                "hessian_eigenvalues": [float(value) for value in eigenvalues],
                "max_hessian_eigenvalue": float(np.max(eigenvalues)),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dps", type=int, default=90)
    parser.add_argument("--steps", default="1e-4,3e-5,1e-5")
    parser.add_argument("--maxiter", type=int, default=20000)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print("k=3 reduced local check")
    print("========================")
    start_time = time.time()
    vector = optimize_reduced(args.maxiter)
    t_star = math.exp(vector[0])
    r_star = math.exp(vector[1])
    theta_star = vector[2] % (2 * math.pi)
    p_star, q_star, h_star = recover_pqh(t_star, r_star, theta_star)
    value_float = reduced_float(vector)
    value_mp = reduced_mp([mp.mpf(str(vector[0])), mp.mpf(str(vector[1])), mp.mpf(str(vector[2]))])
    steps = [float(part) for part in args.steps.split(",") if part.strip()]
    rows = derivative_rows(vector, steps, args.dps)
    elapsed = time.time() - start_time

    print(f"log variables: {vector.tolist()}")
    print(f"(t,r,theta):  {t_star:.17g}  {r_star:.17g}  {theta_star:.17g}")
    print(f"(p,q,h):      {p_star:.17g}  {q_star:.17g}  {h_star:.17g}")
    print(f"value float:  {value_float:.17g}")
    print(f"value mp:     {mp.nstr(value_mp, 80)}")
    for row in rows:
        print(
            f"step={row['step']:.1e} grad_inf={row['gradient_inf']:.3e} "
            f"eig={row['hessian_eigenvalues']}",
            flush=True,
        )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "variables": ["log_t", "log_r", "theta"],
        "log_variables": [float(value) for value in vector],
        "t_r_theta": [t_star, r_star, theta_star],
        "p_q_h": [p_star, q_star, h_star],
        "value_float64": value_float,
        "value_mpmath": str(value_mp),
        "dps": args.dps,
        "derivative_checks": rows,
        "elapsed_seconds": elapsed,
    }
    if not args.no_save:
        output = Path(args.output) if args.output else Path("results") / f"k3_reduced_local_check_{datetime.now():%Y%m%d_%H%M%S}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved: {output}")


if __name__ == "__main__":
    main()