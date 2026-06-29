#!/usr/bin/env python3
"""Cartesian full-block Hessian diagnostic for the k=3 active-set candidate.

The polar/log-amplitude coordinates used by the optimizer are singular at zero
inactive amplitude.  This script rewrites the complete k=3 block in Cartesian
complex coefficients on the divergence-free basis and differentiates -R at the
zero-floor 9-mode candidate.  It is the local full-block exclusion diagnostic:
positive curvature of -R in inactive Cartesian directions means no small
full-block release can increase R.
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

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.gap3.k3_active_hessian_check import refine_active_support  # noqa: E402
from scripts.gap3.k3_active_set_verify import build_problem_scope  # noqa: E402
from scripts.gap3.k3_closed_form_probe import build_prob, support_from_warm  # noqa: E402


def parse_float_list(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def polar_params_to_coefficients(params: np.ndarray) -> np.ndarray:
    theta = params[0::4]
    phi = params[1::4]
    psi = params[2::4]
    radius = np.sqrt(np.exp(params[3::4]))
    c1 = radius * np.cos(theta) * np.exp(1j * phi)
    c2 = radius * np.sin(theta) * np.exp(1j * psi)
    coeffs = np.empty(2 * len(theta), dtype=np.complex128)
    coeffs[0::2] = c1
    coeffs[1::2] = c2
    return coeffs


def coefficients_to_real(coeffs: np.ndarray) -> np.ndarray:
    vector = np.empty(2 * len(coeffs), dtype=np.float64)
    vector[0::2] = coeffs.real
    vector[1::2] = coeffs.imag
    return vector


def real_to_coefficients(vector: np.ndarray) -> np.ndarray:
    return vector[0::2] + 1j * vector[1::2]


def objective_from_real(problem: dict, vector: np.ndarray) -> float:
    coeffs = real_to_coefficients(vector).reshape(problem["N"], 2)
    u_pos = coeffs[:, 0, None] * problem["e1s"] + coeffs[:, 1, None] * problem["e2s"]
    amplitudes = np.sum(np.abs(coeffs) ** 2, axis=1)
    x2 = 2.0 * float(np.dot(problem["k2s"], amplitudes))
    d2 = 2.0 * float(np.dot(problem["k2s"] ** 2, amplitudes))
    if x2 <= 0.0 or d2 <= 0.0:
        return 0.0
    u_raw = np.vstack([u_pos, u_pos.conj()])
    sdu = np.einsum("td,td->t", problem["s_mat"], u_raw[problem["r_idx"]])
    ced = np.einsum("td,td->t", u_raw[problem["ell_idx"]].conj(), u_raw[problem["s_idx"]])
    b_value = float(-np.imag(np.dot(problem["ell2"] * sdu, ced)))
    return b_value / (x2 * math.sqrt(d2))


def embed_active_coefficients(full_problem: dict, support_modes: list[tuple[int, int, int]], support_params: np.ndarray) -> tuple[np.ndarray, list[int]]:
    full_index = {tuple(int(component) for component in wavevector): index for index, wavevector in enumerate(full_problem["wavevecs"])}
    active_coeffs = polar_params_to_coefficients(support_params)
    full_coeffs = np.zeros(2 * full_problem["N"], dtype=np.complex128)
    active_indices = []
    for support_index, mode in enumerate(support_modes):
        index = full_index[tuple(int(component) for component in mode)]
        full_coeffs[2 * index : 2 * index + 2] = active_coeffs[2 * support_index : 2 * support_index + 2]
        active_indices.append(index)
    return coefficients_to_real(full_coeffs), active_indices


def finite_difference_hessian(problem: dict, vector: np.ndarray, step: float) -> np.ndarray:
    dimension = len(vector)
    f0 = -objective_from_real(problem, vector)
    hessian = np.zeros((dimension, dimension), dtype=np.float64)
    for i in range(dimension):
        ei = np.zeros(dimension, dtype=np.float64)
        ei[i] = step
        f_plus = -objective_from_real(problem, vector + ei)
        f_minus = -objective_from_real(problem, vector - ei)
        hessian[i, i] = (f_plus - 2.0 * f0 + f_minus) / (step * step)
        for j in range(i + 1, dimension):
            ej = np.zeros(dimension, dtype=np.float64)
            ej[j] = step
            f_pp = -objective_from_real(problem, vector + ei + ej)
            f_pm = -objective_from_real(problem, vector + ei - ej)
            f_mp = -objective_from_real(problem, vector - ei + ej)
            f_mm = -objective_from_real(problem, vector - ei - ej)
            value = (f_pp - f_pm - f_mp + f_mm) / (4.0 * step * step)
            hessian[i, j] = value
            hessian[j, i] = value
    return 0.5 * (hessian + hessian.T)


def inactive_quadratic_data(problem: dict, active_indices: list[int]) -> dict:
    active_set = set(active_indices)
    inactive_indices = [index for index in range(problem["N"]) if index not in active_set]
    inactive_position = {mode_index: pos for pos, mode_index in enumerate(inactive_indices)}
    triad_mask = []
    for triad_index in range(len(problem["ell_idx"])):
        bases = [
            int(problem["ell_idx"][triad_index]) % problem["N"],
            int(problem["r_idx"][triad_index]) % problem["N"],
            int(problem["s_idx"][triad_index]) % problem["N"],
        ]
        if sum(base not in active_set for base in bases) == 2:
            triad_mask.append(triad_index)
    return {
        "inactive_indices": inactive_indices,
        "inactive_position": inactive_position,
        "triad_indices": np.array(triad_mask, dtype=np.int64),
    }


def inactive_vector_to_full_coeffs(problem: dict, base_coeffs: np.ndarray, inactive_indices: list[int], vector: np.ndarray) -> np.ndarray:
    coeffs = base_coeffs.copy().reshape(problem["N"], 2)
    inactive_coeffs = real_to_coefficients(vector).reshape(len(inactive_indices), 2)
    for local_index, mode_index in enumerate(inactive_indices):
        coeffs[mode_index, :] = inactive_coeffs[local_index, :]
    return coeffs.reshape(2 * problem["N"])


def b_value_for_triads(problem: dict, coeffs_flat: np.ndarray, triad_indices: np.ndarray) -> float:
    coeffs = coeffs_flat.reshape(problem["N"], 2)
    u_pos = coeffs[:, 0, None] * problem["e1s"] + coeffs[:, 1, None] * problem["e2s"]
    u_raw = np.vstack([u_pos, u_pos.conj()])
    ell_idx = problem["ell_idx"][triad_indices]
    r_idx = problem["r_idx"][triad_indices]
    s_idx = problem["s_idx"][triad_indices]
    s_mat = problem["s_mat"][triad_indices]
    ell2 = problem["ell2"][triad_indices]
    sdu = np.einsum("td,td->t", s_mat, u_raw[r_idx])
    ced = np.einsum("td,td->t", u_raw[ell_idx].conj(), u_raw[s_idx])
    return float(-np.imag(np.dot(ell2 * sdu, ced)))


def inactive_second_order_minus_r(problem: dict, base_coeffs: np.ndarray, active_value: float, inactive_data: dict, vector: np.ndarray) -> float:
    inactive_indices = inactive_data["inactive_indices"]
    coeffs = inactive_vector_to_full_coeffs(problem, base_coeffs, inactive_indices, vector)
    b2 = b_value_for_triads(problem, coeffs, inactive_data["triad_indices"])

    base = base_coeffs.reshape(problem["N"], 2)
    amplitudes0 = np.sum(np.abs(base) ** 2, axis=1)
    x20 = 2.0 * float(np.dot(problem["k2s"], amplitudes0))
    d20 = 2.0 * float(np.dot(problem["k2s"] ** 2, amplitudes0))
    den0 = x20 * math.sqrt(d20)

    inactive_coeffs = real_to_coefficients(vector).reshape(len(inactive_indices), 2)
    inactive_amp = np.sum(np.abs(inactive_coeffs) ** 2, axis=1)
    inactive_k2 = problem["k2s"][np.array(inactive_indices, dtype=int)]
    delta_x2 = 2.0 * float(np.dot(inactive_k2, inactive_amp))
    delta_d2 = 2.0 * float(np.dot(inactive_k2 ** 2, inactive_amp))
    return -b2 / den0 + active_value * (delta_x2 / x20 + 0.5 * delta_d2 / d20)


def inactive_quadratic_hessian(problem: dict, base_coeffs: np.ndarray, active_value: float, inactive_data: dict) -> np.ndarray:
    dimension = 4 * len(inactive_data["inactive_indices"])
    hessian = np.zeros((dimension, dimension), dtype=np.float64)
    basis_values = np.zeros(dimension, dtype=np.float64)
    for i in range(dimension):
        vector = np.zeros(dimension, dtype=np.float64)
        vector[i] = 1.0
        basis_values[i] = inactive_second_order_minus_r(problem, base_coeffs, active_value, inactive_data, vector)
        hessian[i, i] = 2.0 * basis_values[i]
    for i in range(dimension):
        vector = np.zeros(dimension, dtype=np.float64)
        vector[i] = 1.0
        for j in range(i + 1, dimension):
            vector[j] = 1.0
            mixed = inactive_second_order_minus_r(problem, base_coeffs, active_value, inactive_data, vector)
            value = mixed - basis_values[i] - basis_values[j]
            hessian[i, j] = value
            hessian[j, i] = value
            vector[j] = 0.0
    return 0.5 * (hessian + hessian.T)


def block_indices(active_indices: list[int], n_modes: int) -> tuple[list[int], list[int]]:
    active_mode_set = set(active_indices)
    active = []
    inactive = []
    for mode_index in range(n_modes):
        target = active if mode_index in active_mode_set else inactive
        target.extend(range(4 * mode_index, 4 * mode_index + 4))
    return active, inactive


def summarize_eigenvalues(values: np.ndarray, tolerance: float) -> dict:
    values = np.sort(values)
    return {
        "count": int(len(values)),
        "positive": int(np.sum(values > tolerance)),
        "flat": int(np.sum(np.abs(values) <= tolerance)),
        "negative": int(np.sum(values < -tolerance)),
        "min": float(values[0]) if len(values) else None,
        "max": float(values[-1]) if len(values) else None,
        "smallest_20": [float(value) for value in values[:20]],
        "largest_20": [float(value) for value in values[-20:]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=float, default=3e-5)
    parser.add_argument("--full-fd", action="store_true", help="also compute the full finite-difference Hessian")
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--active-refine-starts", type=int, default=8)
    parser.add_argument("--active-refine-scales", default="1e-4,1e-3,1e-2,5e-2")
    parser.add_argument("--active-refine-seed", type=int, default=20260530)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    print("k=3 full-block Cartesian Hessian")
    print("=================================")
    start_time = time.time()
    support_indices, support_modes, support_start = support_from_warm()
    support_problem = build_prob(support_modes)
    support_value, support_params, support_grad, refine_rows = refine_active_support(
        support_problem,
        support_start,
        args.active_refine_starts,
        parse_float_list(args.active_refine_scales),
        args.active_refine_seed,
    )
    full_problem = build_problem_scope("full-block")
    vector, active_indices = embed_active_coefficients(full_problem, support_modes, support_params)
    value = objective_from_real(full_problem, vector)
    active_params, inactive_params = block_indices(active_indices, full_problem["N"])

    print(f"full problem: N={full_problem['N']} T={len(full_problem['ell_idx'])}")
    print(f"active modes: {len(active_indices)} inactive modes: {full_problem['N'] - len(active_indices)}")
    print(f"support value: {support_value:.17g}  full Cartesian value: {value:.17g}")
    print(f"support grad max: {support_grad:.3e}  refine tries: {len(refine_rows)}")
    print(f"Hessian dimension: {len(vector)}  step={args.step:g}")

    base_coeffs = real_to_coefficients(vector)
    inactive_data = inactive_quadratic_data(full_problem, active_indices)
    print(f"inactive quadratic triads: {len(inactive_data['triad_indices'])}")
    inactive_hessian = inactive_quadratic_hessian(full_problem, base_coeffs, value, inactive_data)
    eigen_inactive = np.linalg.eigvalsh(inactive_hessian)
    inactive_summary = summarize_eigenvalues(eigen_inactive, args.tolerance)
    all_summary = None
    active_summary = None
    if args.full_fd:
        hessian = finite_difference_hessian(full_problem, vector, args.step)
        eigen_all = np.linalg.eigvalsh(hessian)
        eigen_active = np.linalg.eigvalsh(hessian[np.ix_(active_params, active_params)])
        all_summary = summarize_eigenvalues(eigen_all, args.tolerance)
        active_summary = summarize_eigenvalues(eigen_active, args.tolerance)
        print(
            f"all:      pos={all_summary['positive']} flat={all_summary['flat']} "
            f"neg={all_summary['negative']} min={all_summary['min']:+.3e}"
        )
        print(
            f"active:   pos={active_summary['positive']} flat={active_summary['flat']} "
            f"neg={active_summary['negative']} min={active_summary['min']:+.3e}"
        )
    print(
        f"inactive: pos={inactive_summary['positive']} flat={inactive_summary['flat']} "
        f"neg={inactive_summary['negative']} min={inactive_summary['min']:+.3e}"
    )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "step": args.step,
        "tolerance": args.tolerance,
        "support_indices_from_warm": support_indices,
        "support_modes": support_modes,
        "active_indices_full_block": active_indices,
        "full_modes": int(full_problem["N"]),
        "full_triads": int(len(full_problem["ell_idx"])),
        "support_value_float64": support_value,
        "full_cartesian_value": value,
        "support_grad_max": support_grad,
        "active_refine_top": sorted(refine_rows, key=lambda item: item["value"], reverse=True)[:20],
        "all_hessian": all_summary,
        "active_block_hessian": active_summary,
        "inactive_quadratic_triads": int(len(inactive_data["triad_indices"])),
        "inactive_block_hessian": inactive_summary,
        "elapsed_seconds": time.time() - start_time,
    }
    if not args.no_save:
        output = Path(args.output) if args.output else Path("results") / f"k3_fullblock_cartesian_hessian_{datetime.now():%Y%m%d_%H%M%S}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved: {output}")


if __name__ == "__main__":
    main()