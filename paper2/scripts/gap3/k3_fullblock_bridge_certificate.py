#!/usr/bin/env python3
"""Full-block bridge certificate data for the k=3 active maximizer.

This packages the finite checks that connect the reduced active theorem to the
85-mode full block.  It is an audit artifact: exact combinatorial checks are
kept separate from floating component spectra so the remaining formalization
work is visible.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.gap3.k3_active_hessian_check import refine_active_support  # noqa: E402
from scripts.gap3.k3_active_set_verify import build_problem_scope  # noqa: E402
from scripts.gap3.k3_closed_form_probe import build_prob, support_from_warm  # noqa: E402
from scripts.gap3.k3_fullblock_cartesian_hessian import (  # noqa: E402
    embed_active_coefficients,
    inactive_quadratic_data,
    inactive_quadratic_hessian,
    objective_from_real,
    real_to_coefficients,
)
from scripts.gap3.k3_inactive_release_eigen import release_matrix  # noqa: E402


COORDINATE_LABELS = ("e1_re", "e1_im", "e2_re", "e2_im")


def parse_float_list(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def positive_rep(mode: tuple[int, int, int]) -> tuple[int, int, int]:
    negative = tuple(-component for component in mode)
    return mode if mode > negative else negative


def orbit_supports(problem: dict, active_indices: list[int]) -> list[list[int]]:
    mode_to_index = {tuple(int(component) for component in mode): index for index, mode in enumerate(problem["wavevecs"])}
    canonical = [tuple(int(component) for component in problem["wavevecs"][index]) for index in active_indices]
    supports: set[tuple[int, ...]] = set()
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            image = []
            for mode in canonical:
                mapped = tuple(signs[index] * mode[permutation[index]] for index in range(3))
                image.append(mode_to_index[positive_rep(mapped)])
            supports.add(tuple(sorted(image)))
    return [list(support) for support in sorted(supports)]


def triad_same_inactive_counts(problem: dict, active_set: set[int]) -> list[dict]:
    mode_count = int(problem["N"])
    rows = []
    for mode_index in range(mode_count):
        if mode_index in active_set:
            continue
        count = 0
        examples = []
        for ell_index, right_index, source_index in zip(problem["ell_idx"], problem["r_idx"], problem["s_idx"]):
            bases = [int(ell_index) % mode_count, int(right_index) % mode_count, int(source_index) % mode_count]
            if bases.count(mode_index) == 2 and sum(base in active_set for base in bases) == 1:
                count += 1
                if len(examples) < 5:
                    examples.append([int(ell_index), int(right_index), int(source_index)])
        rows.append(
            {
                "mode_index": int(mode_index),
                "wavevector": [int(component) for component in problem["wavevecs"][mode_index]],
                "same_inactive_active_triad_count": count,
                "examples": examples,
            }
        )
    return rows


def inactive_coordinate(mode_index: int, inactive_indices: list[int]) -> dict:
    inactive_mode_position = mode_index // 4
    coordinate_index = mode_index % 4
    return {
        "inactive_position": inactive_mode_position,
        "mode_index": int(inactive_indices[inactive_mode_position]),
        "coordinate": COORDINATE_LABELS[coordinate_index],
    }


def connected_components(matrix: np.ndarray, threshold: float) -> list[list[int]]:
    dimension = matrix.shape[0]
    adjacency = [set() for _ in range(dimension)]
    row_indices, column_indices = np.nonzero(np.abs(matrix) > threshold)
    for row_index, column_index in zip(row_indices, column_indices):
        if row_index != column_index:
            adjacency[int(row_index)].add(int(column_index))
            adjacency[int(column_index)].add(int(row_index))
    seen = [False] * dimension
    components = []
    for start_index in range(dimension):
        if seen[start_index]:
            continue
        stack = [start_index]
        seen[start_index] = True
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def component_summaries(matrix: np.ndarray, inactive_indices: list[int], threshold: float) -> list[dict]:
    summaries = []
    for component_id, component in enumerate(connected_components(matrix, threshold), 1):
        block = matrix[np.ix_(component, component)]
        eigenvalues = np.linalg.eigvalsh(block)
        diagonal = np.diag(block)
        off_diagonal = np.sum(np.abs(block), axis=1) - np.abs(diagonal)
        involved_modes = sorted({int(inactive_indices[index // 4]) for index in component})
        summaries.append(
            {
                "component_id": component_id,
                "dimension": int(len(component)),
                "inactive_mode_count": int(len(involved_modes)),
                "inactive_modes": involved_modes,
                "coordinates": [inactive_coordinate(index, inactive_indices) for index in component],
                "min_eigenvalue": float(eigenvalues[0]),
                "max_eigenvalue": float(eigenvalues[-1]),
                "min_gershgorin_margin": float(np.min(diagonal - off_diagonal)),
            }
        )
    return sorted(summaries, key=lambda item: item["min_eigenvalue"])


def sparse_matrix_entries(matrix: np.ndarray, threshold: float) -> list[dict]:
    entries = []
    dimension = matrix.shape[0]
    for row_index in range(dimension):
        for column_index in range(row_index, dimension):
            value = float(matrix[row_index, column_index])
            if row_index == column_index or abs(value) > threshold:
                entries.append({"row": row_index, "column": column_index, "value": value})
    return entries


def release_matrix_summary(problem: dict, base_raw: np.ndarray, base_b: float, active_set: set[int], denominator: float) -> dict:
    rows = []
    for mode_index in range(int(problem["N"])):
        if mode_index in active_set:
            continue
        matrix = release_matrix(problem, base_raw, base_b, mode_index)
        row_sum = float(np.max(np.sum(np.abs(matrix), axis=1)))
        max_abs = float(np.max(np.abs(matrix)))
        rows.append(
            {
                "mode_index": int(mode_index),
                "wavevector": [int(component) for component in problem["wavevecs"][mode_index]],
                "q_matrix_max_abs": max_abs,
                "q_matrix_row_sum_bound": row_sum,
                "q_over_denominator_row_sum_bound": row_sum / denominator,
            }
        )
    rows.sort(key=lambda item: item["q_matrix_row_sum_bound"], reverse=True)
    return {
        "max_abs_entry": rows[0]["q_matrix_max_abs"] if rows else 0.0,
        "max_row_sum": rows[0]["q_matrix_row_sum_bound"] if rows else 0.0,
        "worst_rows": rows[:12],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-refine-starts", type=int, default=0)
    parser.add_argument("--active-refine-scales", default="1e-4,1e-3,1e-2,5e-2")
    parser.add_argument("--active-refine-seed", type=int, default=20260530)
    parser.add_argument("--zero-threshold", type=float, default=1e-14)
    parser.add_argument("--include-matrix", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

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
    problem = build_problem_scope("full-block")
    active_vector, active_indices = embed_active_coefficients(problem, support_modes, support_params)
    active_set = set(active_indices)
    full_value = objective_from_real(problem, active_vector)
    base_coefficients = real_to_coefficients(active_vector)
    inactive_data = inactive_quadratic_data(problem, active_indices)
    inactive_indices = [int(index) for index in inactive_data["inactive_indices"]]
    inactive_hessian = inactive_quadratic_hessian(problem, base_coefficients, full_value, inactive_data)
    inactive_eigenvalues = np.linalg.eigvalsh(inactive_hessian)
    components = component_summaries(inactive_hessian, inactive_indices, args.zero_threshold)

    amplitudes = np.sum(np.abs(base_coefficients.reshape(problem["N"], 2)) ** 2, axis=1)
    x_squared = 2.0 * float(np.dot(problem["k2s"], amplitudes))
    d_squared = 2.0 * float(np.dot(problem["k2s"] ** 2, amplitudes))
    denominator = x_squared * math.sqrt(d_squared)
    base_raw = np.vstack(
        [
            base_coefficients.reshape(problem["N"], 2)[:, 0, None] * problem["e1s"]
            + base_coefficients.reshape(problem["N"], 2)[:, 1, None] * problem["e2s"],
            (
                base_coefficients.reshape(problem["N"], 2)[:, 0, None] * problem["e1s"]
                + base_coefficients.reshape(problem["N"], 2)[:, 1, None] * problem["e2s"]
            ).conj(),
        ]
    )
    active_b = full_value * denominator
    same_mode_rows = triad_same_inactive_counts(problem, active_set)
    same_mode_max = max(row["same_inactive_active_triad_count"] for row in same_mode_rows)
    support_orbit = orbit_supports(problem, active_indices)

    print("k=3 full-block bridge certificate")
    print("==================================")
    print(f"full block: modes={problem['N']} triads={len(problem['ell_idx'])}")
    print(f"active modes={len(active_indices)} inactive modes={len(inactive_indices)} orbit supports={len(support_orbit)}")
    print(f"active value={full_value:.17g} support value={support_value:.17g} support grad={support_grad:.3e}")
    print(f"same-inactive quadratic triads: max={same_mode_max}")
    print(
        f"inactive Hessian: dim={inactive_hessian.shape[0]} min_eig={inactive_eigenvalues[0]:+.12e} "
        f"components={len(components)} sizes={dict(Counter(item['dimension'] for item in components))}"
    )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "audit_certificate_not_formal_interval_proof",
        "support_indices_from_warm": support_indices,
        "support_modes": support_modes,
        "active_indices_full_block": [int(index) for index in active_indices],
        "full_modes": int(problem["N"]),
        "full_triads": int(len(problem["ell_idx"])),
        "active_modes": int(len(active_indices)),
        "inactive_modes": int(len(inactive_indices)),
        "orbit_support_count": int(len(support_orbit)),
        "orbit_supports": support_orbit,
        "support_value_float64": support_value,
        "full_cartesian_value": full_value,
        "support_grad_max": support_grad,
        "active_refine_starts_per_scale": args.active_refine_starts,
        "active_refine_top": sorted(refine_rows, key=lambda item: item["value"], reverse=True)[:20],
        "X2": x_squared,
        "D2": d_squared,
        "denominator": denominator,
        "B": active_b,
        "same_inactive_active_triad_count_max": int(same_mode_max),
        "same_inactive_active_triad_rows_nonzero": [row for row in same_mode_rows if row["same_inactive_active_triad_count"]],
        "one_mode_release_q_matrix_float_sanity": release_matrix_summary(problem, base_raw, active_b, active_set, denominator),
        "inactive_quadratic_triads": int(len(inactive_data["triad_indices"])),
        "inactive_hessian_dimension": int(inactive_hessian.shape[0]),
        "inactive_hessian_min_eigenvalue": float(inactive_eigenvalues[0]),
        "inactive_hessian_max_eigenvalue": float(inactive_eigenvalues[-1]),
        "inactive_component_count": int(len(components)),
        "inactive_component_size_counts": dict(Counter(item["dimension"] for item in components)),
        "inactive_components_by_min_eigenvalue": components,
        "zero_threshold": args.zero_threshold,
        "elapsed_seconds": time.time() - start_time,
    }
    if args.include_matrix:
        summary["inactive_hessian_sparse_entries"] = sparse_matrix_entries(inactive_hessian, args.zero_threshold)

    if not args.no_save:
        output = Path(args.output) if args.output else Path("results") / f"k3_fullblock_bridge_certificate_{datetime.now():%Y%m%d_%H%M%S}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved: {output}")


if __name__ == "__main__":
    main()