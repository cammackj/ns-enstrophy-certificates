#!/usr/bin/env python3
"""Verify the k=3 closed-form active-set candidate against the nucleus or full block.

This is a research verifier, not a manuscript certificate.  It checks whether
the nine-mode reduced candidate survives when embedded into the full k=3
DCxA nucleus or the full k=3 block, and whether the old scan value is
explained by the inactive-mode floor used by the certifier.
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

from scripts.gap3.gap3_principled_scan import (  # noqa: E402
    _build_problem,
    _restrict_to_active_modes,
    find_dcxa_nucleus,
)
from scripts.gap3.multi_mode_beta_bound import divfree_basis, get_wavevectors, precompute_triads  # noqa: E402
from scripts.gap3.k3_closed_form_probe import (  # noqa: E402
    build_prob,
    mpmath_value,
    support_from_warm,
)
from scripts.gap3.k3_active_hessian_check import refine_active_support  # noqa: E402
from scripts.gap3.max_b_over_keff import neg_ratio_and_grad  # noqa: E402


def parse_float_list(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def build_full_nucleus_problem() -> dict:
    n_d, n_c, n_a = find_dcxa_nucleus(3)
    return _restrict_to_active_modes(_build_problem([n_d, n_c, n_a]))


def build_full_block_problem() -> dict:
    wavevectors = get_wavevectors(max_shell2=2**4 - 1, min_shell2=2**3)
    _, ell_idx, ell2, r_idx, s_idx, s_mat = precompute_triads(wavevectors)
    return {
        "N": len(wavevectors),
        "k2s": np.array([sum(component * component for component in wavevector) for wavevector in wavevectors], dtype=float),
        "e1s": np.array([divfree_basis(wavevector)[0] for wavevector in wavevectors]),
        "e2s": np.array([divfree_basis(wavevector)[1] for wavevector in wavevectors]),
        "ell_idx": ell_idx,
        "ell2": ell2,
        "r_idx": r_idx,
        "s_idx": s_idx,
        "s_mat": s_mat,
        "wavevecs": wavevectors,
    }


def build_problem_scope(scope: str) -> dict:
    if scope == "nucleus":
        return build_full_nucleus_problem()
    if scope == "full-block":
        return build_full_block_problem()
    raise ValueError(f"unknown scope {scope!r}")


def ratio_and_grad(problem: dict, params: np.ndarray) -> tuple[float, np.ndarray]:
    value, grad = neg_ratio_and_grad(
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
    return -float(value), grad


def make_bounds(n_modes: int, floor: float) -> list[tuple[float, float]]:
    return [(0.0, math.pi / 2), (0.0, 2 * math.pi), (0.0, 2 * math.pi), (floor, 8.0)] * n_modes


def polish(problem: dict, params: np.ndarray, floor: float, gtol: float, maxiter: int) -> tuple[float, np.ndarray, float]:
    bounds = make_bounds(problem["N"], floor)

    def objective(vector: np.ndarray):
        return neg_ratio_and_grad(
            vector,
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

    result = minimize(
        objective,
        params,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"ftol": 1e-16, "gtol": gtol, "maxiter": maxiter},
    )
    return -float(result.fun), result.x, float(np.max(np.abs(result.jac)))


def embed_support(
    full_problem: dict,
    support_modes: list[tuple[int, int, int]],
    support_params: np.ndarray,
    floor: float,
) -> tuple[np.ndarray, list[int]]:
    n_modes = full_problem["N"]
    embedded = np.zeros(4 * n_modes)
    embedded[3::4] = floor
    mode_to_support = {tuple(mode): index for index, mode in enumerate(support_modes)}
    active_indices: list[int] = []
    for index, wavevector in enumerate(full_problem["wavevecs"]):
        key = tuple(int(component) for component in wavevector)
        if key in mode_to_support:
            support_index = mode_to_support[key]
            embedded[4 * index : 4 * index + 4] = support_params[4 * support_index : 4 * support_index + 4]
            active_indices.append(index)
    return embedded, active_indices


def projected_grad_norm(params: np.ndarray, grad: np.ndarray, floor: float) -> float:
    projected = np.abs(grad.copy())
    n_modes = len(params) // 4
    for index in range(n_modes):
        if params[4 * index + 3] <= floor + 1e-6:
            projected[4 * index : 4 * index + 3] = 0.0
            if grad[4 * index + 3] > 0:
                projected[4 * index + 3] = 0.0
    return float(np.max(projected))


def inactive_release_summary(
    problem: dict,
    params: np.ndarray,
    grad: np.ndarray,
    active_indices: list[int],
    tolerance: float = 1e-10,
) -> dict:
    """Summarize one-sided inactive-mode release gradients.

    The optimizer uses log-amplitudes.  At an inactive floor the raw loga
    gradient scales like exp(loga), so the meaningful first-order release
    coefficient is d(-R)/d(exp(loga)) = grad_loga / exp(loga).  Positive values
    mean releasing that inactive amplitude decreases R to first order.
    """
    active_set = set(active_indices)
    rows = []
    for index, wavevector in enumerate(problem["wavevecs"]):
        if index in active_set:
            continue
        loga = float(params[4 * index + 3])
        amplitude = math.exp(loga)
        raw = float(grad[4 * index + 3])
        scaled = raw / amplitude if amplitude > 0.0 else math.inf
        rows.append(
            {
                "mode_index": index,
                "shell": int(problem["k2s"][index]),
                "wavevector": tuple(int(component) for component in wavevector),
                "raw_log_amplitude_gradient": raw,
                "scaled_amplitude_gradient": scaled,
            }
        )
    rows.sort(key=lambda item: abs(item["scaled_amplitude_gradient"]), reverse=True)
    scaled_values = [item["scaled_amplitude_gradient"] for item in rows]
    negative = [item for item in rows if item["scaled_amplitude_gradient"] < -tolerance]
    return {
        "inactive_modes": len(rows),
        "scaled_min": min(scaled_values) if scaled_values else None,
        "scaled_max": max(scaled_values) if scaled_values else None,
        "scaled_negative_count": len(negative),
        "scaled_tolerance": tolerance,
        "top_abs_scaled": rows[:12],
    }


def floor_sweep(
    full_problem: dict,
    support_modes: list[tuple[int, int, int]],
    support_params: np.ndarray,
    target_value: float,
    floors: list[float],
    gtol: float,
    maxiter: int,
) -> list[dict]:
    rows = []
    for floor in floors:
        embedded, active_indices = embed_support(full_problem, support_modes, support_params, floor)
        embedded_value, embedded_grad = ratio_and_grad(full_problem, embedded)
        polished_value, polished_params, polished_grad = polish(full_problem, embedded, floor, gtol, maxiter)
        _, grad_after = ratio_and_grad(full_problem, polished_params)
        inactive_summary = inactive_release_summary(full_problem, polished_params, grad_after, active_indices)
        row = {
            "floor": floor,
            "embedded_value": embedded_value,
            "embedded_projected_grad": projected_grad_norm(embedded, embedded_grad, floor),
            "polished_value": polished_value,
            "gap_to_target": target_value - polished_value,
            "polished_grad_max": polished_grad,
            "active_modes_after_polish": int(np.sum(polished_params[3::4] > floor + 1e-6)),
            "mapped_active_modes": len(active_indices),
            "inactive_release_summary": inactive_summary,
        }
        rows.append(row)
        print(
            f"floor {floor:>6g}: embedded={embedded_value:.17g} "
            f"polished={polished_value:.17g} gap={target_value - polished_value:+.3e} "
            f"active={row['active_modes_after_polish']} "
            f"inactive_scaled_min={inactive_summary['scaled_min']:+.3e}",
            flush=True,
        )
    return rows


def basin_scan(
    full_problem: dict,
    base_params: np.ndarray,
    active_indices: list[int],
    target_value: float,
    floor: float,
    active_scales: list[float],
    inactive_scales: list[float],
    active_starts: int,
    inactive_starts: int,
    seed: int,
    gtol: float,
    maxiter: int,
) -> dict:
    rng = np.random.default_rng(seed)
    bounds = make_bounds(full_problem["N"], floor)
    lo = np.array([bound[0] for bound in bounds])
    hi = np.array([bound[1] for bound in bounds])

    starts: list[tuple[str, np.ndarray]] = []
    for scale in active_scales:
        for _ in range(active_starts):
            params = base_params.copy()
            for index in active_indices:
                params[4 * index : 4 * index + 3] += rng.normal(0.0, scale, size=3)
                params[4 * index + 3] += rng.normal(0.0, scale)
            starts.append((f"active_noise={scale:g}", np.minimum(np.maximum(params, lo), hi)))

    active_set = set(active_indices)
    for scale in inactive_scales:
        for _ in range(inactive_starts):
            params = base_params.copy()
            for index in range(full_problem["N"]):
                params[4 * index : 4 * index + 3] = rng.uniform(
                    [0.0, 0.0, 0.0], [math.pi / 2, 2 * math.pi, 2 * math.pi]
                )
                if index in active_set:
                    params[4 * index + 3] += rng.normal(0.0, scale)
                else:
                    params[4 * index + 3] = floor + rng.uniform(0.0, 5.0)
            starts.append((f"inactive_release={scale:g}", np.minimum(np.maximum(params, lo), hi)))

    values = []
    hits_1e10 = 0
    hits_1e8 = 0
    start_time = time.time()
    for index, (label, start) in enumerate(starts, 1):
        value, params, grad_max = polish(full_problem, start, floor, gtol, maxiter)
        active_count = int(np.sum(params[3::4] > floor + 1e-6))
        gap = target_value - value
        if abs(gap) < 1e-10:
            hits_1e10 += 1
        if abs(gap) < 1e-8:
            hits_1e8 += 1
        values.append(
            {
                "label": label,
                "value": value,
                "gap_to_target": gap,
                "grad_max": grad_max,
                "active_modes": active_count,
            }
        )
        print(
            f"{index:02d}/{len(starts)} {label:22s} value={value:.17g} "
            f"gap={gap:+.3e} grad={grad_max:.2e} active={active_count}",
            flush=True,
        )

    values_sorted = sorted(values, key=lambda item: item["value"], reverse=True)
    return {
        "floor": floor,
        "seed": seed,
        "starts": len(starts),
        "hits_within_1e-10": hits_1e10,
        "hits_within_1e-8": hits_1e8,
        "best_value": values_sorted[0]["value"] if values_sorted else None,
        "worst_value": values_sorted[-1]["value"] if values_sorted else None,
        "best_gap_to_target": target_value - values_sorted[0]["value"] if values_sorted else None,
        "worst_gap_to_target": target_value - values_sorted[-1]["value"] if values_sorted else None,
        "elapsed_seconds": time.time() - start_time,
        "top_values": values_sorted[:10],
        "all_values": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floors", default="-8,-10,-12,-16,-20,-24,-30")
    parser.add_argument("--basin-floor", type=float, default=-30.0)
    parser.add_argument("--active-scales", default="1e-4,1e-3,1e-2,5e-2,1e-1")
    parser.add_argument("--inactive-scales", default="1e-3,1e-2,5e-2")
    parser.add_argument("--active-starts", type=int, default=8)
    parser.add_argument("--inactive-starts", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--support-refine-starts", type=int, default=0)
    parser.add_argument("--support-refine-scales", default="1e-4,1e-3,1e-2,5e-2")
    parser.add_argument("--support-refine-seed", type=int, default=20260530)
    parser.add_argument("--scope", choices=("nucleus", "full-block"), default="nucleus")
    parser.add_argument("--gtol", type=float, default=1e-12)
    parser.add_argument("--maxiter", type=int, default=30000)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print("k=3 active-set verification")
    print("===========================")
    support_indices, support_modes, support_start = support_from_warm()
    support_problem = build_prob(support_modes)
    support_value, support_params, support_grad, refine_rows = refine_active_support(
        support_problem,
        support_start,
        args.support_refine_starts,
        parse_float_list(args.support_refine_scales),
        args.support_refine_seed,
    )
    support_mp = mpmath_value(support_problem, support_params, dps=90)
    full_problem = build_problem_scope(args.scope)

    print(f"support indices: {support_indices}")
    print(f"support modes:   {len(support_modes)}")
    print(f"support triads:  {len(support_problem['ell_idx'])}")
    print(f"scope:           {args.scope}")
    print(f"problem:         N={full_problem['N']} T={len(full_problem['ell_idx'])}")
    print(f"support value:   {support_value:.17g}")
    print(f"support mpmath:  {support_mp}")
    print(f"support grad:    {support_grad:.3e}")
    print(f"support refine tries: {len(refine_rows)}")

    floors = parse_float_list(args.floors)
    floor_rows = floor_sweep(
        full_problem,
        support_modes,
        support_params,
        support_value,
        floors,
        args.gtol,
        args.maxiter,
    )

    base_params, active_indices = embed_support(full_problem, support_modes, support_params, args.basin_floor)
    basin = basin_scan(
        full_problem,
        base_params,
        active_indices,
        support_value,
        args.basin_floor,
        parse_float_list(args.active_scales),
        parse_float_list(args.inactive_scales),
        args.active_starts,
        args.inactive_starts,
        args.seed,
        args.gtol,
        args.maxiter,
    )

    print("\nSummary")
    print("-------")
    print(f"target support value: {support_value:.17g}")
    print(
        f"basin starts: {basin['starts']}  hits<1e-10: {basin['hits_within_1e-10']}  "
        f"hits<1e-8: {basin['hits_within_1e-8']}"
    )
    print(f"best basin value: {basin['best_value']:.17g}  gap={basin['best_gap_to_target']:+.3e}")
    print(f"worst basin value: {basin['worst_value']:.17g}  gap={basin['worst_gap_to_target']:+.3e}")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "support_indices": support_indices,
        "support_modes": support_modes,
        "support_value_float64": support_value,
        "support_value_mpmath": str(support_mp),
        "support_grad_max": support_grad,
        "support_refine_starts_per_scale": args.support_refine_starts,
        "support_refine_scales": parse_float_list(args.support_refine_scales),
        "support_refine_top": sorted(refine_rows, key=lambda item: item["value"], reverse=True)[:20],
        "scope": args.scope,
        "problem_modes": int(full_problem["N"]),
        "problem_triads": int(len(full_problem["ell_idx"])),
        "floor_sweep": floor_rows,
        "basin_scan": basin,
    }

    if not args.no_save:
        output = Path(args.output) if args.output else Path("results") / f"k3_active_set_verify_{datetime.now():%Y%m%d_%H%M%S}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved: {output}")


if __name__ == "__main__":
    main()