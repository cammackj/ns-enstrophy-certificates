#!/usr/bin/env python3
"""Scan the strengthened k=3 full-block inequality R + mu E_perp <= C3.

Here E_perp is the X-energy fraction outside the nine active modes.  A proof of
this penalized inequality would immediately give the full-block global exclusion
and equality rigidity: any field with inactive energy has R < C3.
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
from scipy.optimize import minimize

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.gap3.k3_active_set_verify import build_problem_scope  # noqa: E402
from scripts.gap3.k3_closed_form_probe import support_from_warm  # noqa: E402
from scripts.gap3.max_b_over_keff import neg_ratio_and_grad  # noqa: E402


C3_TARGET = 0.021936469459403747249299192478957700397867315103825


def active_indices(problem: dict) -> list[int]:
    _, support_modes, _ = support_from_warm()
    index = {tuple(mode): mode_index for mode_index, mode in enumerate(problem["wavevecs"])}
    return [index[tuple(mode)] for mode in support_modes]


def active_warm_start(problem: dict) -> np.ndarray:
    _, support_modes, support_params = support_from_warm()
    index = {tuple(mode): mode_index for mode_index, mode in enumerate(problem["wavevecs"])}
    params = np.empty(4 * problem["N"], dtype=float)
    params[0::4] = np.pi / 4.0
    params[1::4] = 0.0
    params[2::4] = 0.0
    params[3::4] = -32.0
    for support_index, mode in enumerate(support_modes):
        mode_index = index[tuple(mode)]
        params[4 * mode_index : 4 * mode_index + 4] = support_params[4 * support_index : 4 * support_index + 4]
    return params


def random_params(rng: np.random.Generator, modes: int) -> np.ndarray:
    params = np.empty(4 * modes, dtype=float)
    params[0::4] = rng.uniform(0.0, np.pi / 2.0, size=modes)
    params[1::4] = rng.uniform(0.0, 2.0 * np.pi, size=modes)
    params[2::4] = rng.uniform(0.0, 2.0 * np.pi, size=modes)
    params[3::4] = rng.normal(0.0, 1.0, size=modes)
    return params


def e_perp_and_grad(problem: dict, params: np.ndarray, inactive_mask: np.ndarray) -> tuple[float, np.ndarray]:
    amplitudes = np.exp(params[3::4])
    x_weights = 2.0 * problem["k2s"] * amplitudes
    x_total = float(np.sum(x_weights))
    x_perp = float(np.sum(x_weights[inactive_mask]))
    fraction = x_perp / x_total
    gradient = np.zeros_like(params)
    if x_total <= 0.0:
        return 0.0, gradient
    d_fraction = np.where(inactive_mask, x_weights * (x_total - x_perp) / (x_total * x_total), -x_weights * x_perp / (x_total * x_total))
    gradient[3::4] = d_fraction
    return fraction, gradient


def penalized_negative(problem: dict, inactive_mask: np.ndarray, mu: float, params: np.ndarray) -> tuple[float, np.ndarray, float, float]:
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
    e_perp, grad_e_perp = e_perp_and_grad(problem, params, inactive_mask)
    value = negative_r - mu * e_perp
    gradient = np.asarray(grad_negative_r, dtype=float) - mu * grad_e_perp
    return float(value), gradient, -float(negative_r), e_perp


def polish(problem: dict, inactive_mask: np.ndarray, mu: float, start: np.ndarray, maxiter: int) -> dict:
    bounds = [(0.0, np.pi / 2.0), (0.0, 2.0 * np.pi), (0.0, 2.0 * np.pi), (-40.0, 40.0)] * problem["N"]
    last = {"r": None, "e_perp": None}

    def fun(params: np.ndarray) -> float:
        value, _, r_value, e_perp = penalized_negative(problem, inactive_mask, mu, params)
        last["r"] = r_value
        last["e_perp"] = e_perp
        return value

    def jac(params: np.ndarray) -> np.ndarray:
        return penalized_negative(problem, inactive_mask, mu, params)[1]

    result = minimize(fun, start, jac=jac, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter, "gtol": 1e-10, "ftol": 1e-14})
    value, gradient, r_value, e_perp = penalized_negative(problem, inactive_mask, mu, np.asarray(result.x, dtype=float))
    penalized = -value
    return {
        "penalized_value": penalized,
        "r_value": r_value,
        "e_perp": e_perp,
        "gap_to_target": C3_TARGET - penalized,
        "gradient_max_abs": float(np.max(np.abs(gradient))),
        "active_modes_floor20": int(np.sum(result.x[3::4] > -20.0)),
        "success": bool(result.success),
        "message": str(result.message),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mu", type=float, default=1e-3)
    parser.add_argument("--starts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--maxiter", type=int, default=2000)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    start_time = time.time()
    problem = build_problem_scope("full-block")
    active = set(active_indices(problem))
    inactive_mask = np.array([index not in active for index in range(problem["N"])], dtype=bool)
    starts = [active_warm_start(problem)]
    rng = np.random.default_rng(args.seed)
    starts.extend(random_params(rng, problem["N"]) for _ in range(args.starts))

    print(f"k=3 penalized full-block scan: mu={args.mu:g} starts={len(starts)}")
    print(f"full problem: N={problem['N']} T={len(problem['ell_idx'])} inactive={int(np.sum(inactive_mask))}")
    rows = []
    for index, start in enumerate(starts, start=1):
        row = polish(problem, inactive_mask, args.mu, start, args.maxiter)
        row["start_index"] = index
        rows.append(row)
        print(
            f"{index:3d}/{len(starts)} penalized={row['penalized_value']:.15f} "
            f"R={row['r_value']:.15f} Eperp={row['e_perp']:.3e} "
            f"gap={row['gap_to_target']:.3e} grad={row['gradient_max_abs']:.2e} active={row['active_modes_floor20']}",
            flush=True,
        )
    rows.sort(key=lambda item: item["penalized_value"], reverse=True)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mu": args.mu,
        "target": C3_TARGET,
        "full_modes": int(problem["N"]),
        "full_triads": int(len(problem["ell_idx"])),
        "inactive_modes": int(np.sum(inactive_mask)),
        "rows": rows,
        "elapsed_seconds": time.time() - start_time,
    }
    if not args.no_save:
        output = Path(args.output) if args.output else Path("results") / f"k3_penalized_fullblock_scan_{datetime.now():%Y%m%d_%H%M%S}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved: {output}")


if __name__ == "__main__":
    main()