#!/usr/bin/env python3
"""Exact quadratic inactive-release eigencheck for the k=3 active candidate.

At the zero-inactive active candidate, releasing one inactive positive mode with
complex polarization v and amplitude sqrt(a) gives

    B(u + sqrt(a)v) = B(u) + a Q_j(v) + O(a^{3/2}).

The denominator derivative is independent of the polarization.  Therefore the
minimum one-sided release coefficient d(-R)/da over all polarizations is the
constant denominator penalty minus the largest eigenvalue of the real quadratic
form Q_j(v), where v has four real coordinates in the two complex divergence-
free basis directions.

This replaces angular sampling by a finite eigenvalue computation for each
inactive mode.  The output is still floating point; the matrices are small
enough to interval/rationalize later.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.gap3.k3_active_hessian_check import refine_active_support  # noqa: E402
from scripts.gap3.k3_active_set_verify import build_problem_scope, embed_support  # noqa: E402
from scripts.gap3.k3_closed_form_probe import build_prob, support_from_warm  # noqa: E402


def parse_float_list(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def params_to_u_pos(problem: dict, params: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    theta = params[0::4][:, None]
    phi = params[1::4][:, None]
    psi = params[2::4][:, None]
    amp = np.exp(params[3::4])
    radius = np.sqrt(amp)[:, None]
    u_pos = radius * (
        np.cos(theta) * np.exp(1j * phi) * problem["e1s"]
        + np.sin(theta) * np.exp(1j * psi) * problem["e2s"]
    )
    x2 = 2.0 * float(np.dot(problem["k2s"], amp))
    d2 = 2.0 * float(np.dot(problem["k2s"] * problem["k2s"], amp))
    u_raw = np.vstack([u_pos, u_pos.conj()])
    b_value = compute_b(problem, u_raw)
    return u_pos, u_raw, x2, d2, b_value


def compute_b(problem: dict, u_raw: np.ndarray) -> float:
    sdu = np.einsum("td,td->t", problem["s_mat"], u_raw[problem["r_idx"]])
    ced = np.einsum("td,td->t", np.conjugate(u_raw[problem["ell_idx"]]), u_raw[problem["s_idx"]])
    return float(-np.imag(np.dot(problem["ell2"] * sdu, ced)))


def coordinate_vector(problem: dict, mode_index: int, coord_index: int) -> np.ndarray:
    n_modes = int(problem["N"])
    u_raw = np.zeros((2 * n_modes, 3), dtype=np.complex128)
    polarization = coord_index // 2
    part = coord_index % 2
    basis = problem["e1s"][mode_index] if polarization == 0 else problem["e2s"][mode_index]
    coefficient = 1.0 if part == 0 else 1.0j
    u_raw[mode_index, :] = coefficient * basis
    u_raw[mode_index + n_modes, :] = np.conjugate(coefficient * basis)
    return u_raw


def quadratic_coefficient(problem: dict, base_raw: np.ndarray, b0: float, perturbation: np.ndarray) -> float:
    plus = compute_b(problem, base_raw + perturbation)
    minus = compute_b(problem, base_raw - perturbation)
    return 0.5 * (plus + minus - 2.0 * b0)


def release_matrix(problem: dict, base_raw: np.ndarray, b0: float, mode_index: int) -> np.ndarray:
    basis = [coordinate_vector(problem, mode_index, coord_index) for coord_index in range(4)]
    matrix = np.zeros((4, 4), dtype=float)
    diag = []
    for coord_index, vector in enumerate(basis):
        value = quadratic_coefficient(problem, base_raw, b0, vector)
        matrix[coord_index, coord_index] = value
        diag.append(value)
    for left in range(4):
        for right in range(left + 1, 4):
            value = quadratic_coefficient(problem, base_raw, b0, basis[left] + basis[right])
            matrix[left, right] = matrix[right, left] = 0.5 * (value - diag[left] - diag[right])
    return 0.5 * (matrix + matrix.T)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("nucleus", "full-block"), default="full-block")
    parser.add_argument("--active-refine-starts", type=int, default=24)
    parser.add_argument("--active-refine-scales", default="1e-4,1e-3,1e-2,5e-2")
    parser.add_argument("--active-refine-seed", type=int, default=20260530)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    support_indices, support_modes, support_start = support_from_warm()
    support_problem = build_prob(support_modes)
    support_value, support_params, support_grad, refine_rows = refine_active_support(
        support_problem,
        support_start,
        args.active_refine_starts,
        parse_float_list(args.active_refine_scales),
        args.active_refine_seed,
    )

    problem = build_problem_scope(args.scope)
    base_params, active_indices = embed_support(problem, support_modes, support_params, -800.0)
    active_set = set(active_indices)
    # Replace underflowed inactive parameters by exact zero in the raw field.
    _, base_raw, x2, d2, b0 = params_to_u_pos(problem, base_params)
    for mode_index in range(problem["N"]):
        if mode_index not in active_set:
            base_raw[mode_index, :] = 0.0
            base_raw[mode_index + problem["N"], :] = 0.0
    b0 = compute_b(problem, base_raw)
    d_value = math.sqrt(d2)
    r_value = b0 / (x2 * d_value)

    print("k=3 inactive release eigencheck")
    print("=================================")
    print(f"scope: {args.scope}  N={problem['N']} T={len(problem['ell_idx'])}")
    print(f"active value: {r_value:.17g}  support refine value={support_value:.17g} grad={support_grad:.3e}")

    rows = []
    for mode_index in range(problem["N"]):
        if mode_index in active_set:
            continue
        matrix = release_matrix(problem, base_raw, b0, mode_index)
        eigenvalues = np.linalg.eigvalsh(matrix)
        max_q = float(eigenvalues[-1])
        shell = float(problem["k2s"][mode_index])
        denominator_derivative = 2.0 * shell * d_value + x2 * shell * shell / d_value
        denominator_penalty = r_value * denominator_derivative / (x2 * d_value)
        min_release = denominator_penalty - max_q / (x2 * d_value)
        row = {
            "mode_index": int(mode_index),
            "shell": int(shell),
            "wavevector": [int(component) for component in problem["wavevecs"][mode_index]],
            "min_release_coefficient": min_release,
            "denominator_penalty": denominator_penalty,
            "max_q_over_unit_polarization": max_q,
            "q_eigenvalues": [float(item) for item in eigenvalues],
            "q_matrix": [[float(item) for item in line] for line in matrix],
        }
        rows.append(row)
        print(
            f"idx={mode_index:2d} shell={int(shell):2d} mode={tuple(row['wavevector'])} "
            f"min_release={min_release:+.12e} max_q={max_q:+.12e}",
            flush=True,
        )

    rows.sort(key=lambda item: item["min_release_coefficient"])
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scope": args.scope,
        "problem_modes": int(problem["N"]),
        "problem_triads": int(len(problem["ell_idx"])),
        "support_indices": support_indices,
        "support_modes": support_modes,
        "support_value_float64": support_value,
        "support_grad_max": support_grad,
        "active_refine_top": sorted(refine_rows, key=lambda item: item["value"], reverse=True)[:20],
        "active_indices": sorted(int(index) for index in active_set),
        "base_value": r_value,
        "X2": x2,
        "D2": d2,
        "B": b0,
        "minimum": rows[0],
        "negative_count_below_1e-12": sum(1 for row in rows if row["min_release_coefficient"] < -1e-12),
        "inactive_modes": rows,
    }
    print("\nSummary")
    print("-------")
    print(
        f"minimum release coefficient: {rows[0]['min_release_coefficient']:+.12e} "
        f"at idx={rows[0]['mode_index']} mode={tuple(rows[0]['wavevector'])}"
    )
    print(f"negative coefficients below -1e-12: {summary['negative_count_below_1e-12']}")
    if not args.no_save:
        output = Path(args.output) if args.output else Path("results") / f"k3_inactive_release_eigen_{datetime.now():%Y%m%d_%H%M%S}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved: {output}")


if __name__ == "__main__":
    main()