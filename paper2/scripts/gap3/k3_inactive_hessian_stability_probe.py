#!/usr/bin/env python3
"""Probe stability of the k=3 inactive Hessian under active perturbations.

This is a sizing tool for the local full-block bridge.  It does not claim an
interval proof; it estimates how large a neighborhood around the active point
can be handled by the inactive Hessian margin.
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


def connected_components(matrix: np.ndarray, threshold: float) -> list[list[int]]:
    dimension = matrix.shape[0]
    adjacency = [set() for _ in range(dimension)]
    row_indices, column_indices = np.nonzero(np.abs(matrix) > threshold)
    for row_index, column_index in zip(row_indices, column_indices):
        if row_index != column_index:
            adjacency[int(row_index)].add(int(column_index))
            adjacency[int(column_index)].add(int(row_index))
    seen = [False] * dimension
    components: list[list[int]] = []
    for start in range(dimension):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
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


def parse_float_list(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def inactive_hessian_for_params(full_problem: dict, support_modes: list[tuple[int, int, int]], support_params: np.ndarray) -> tuple[float, np.ndarray]:
    vector, active_indices = embed_active_coefficients(full_problem, support_modes, support_params)
    value = objective_from_real(full_problem, vector)
    base_coefficients = real_to_coefficients(vector)
    inactive_data = inactive_quadratic_data(full_problem, active_indices)
    matrix = inactive_quadratic_hessian(full_problem, base_coefficients, value, inactive_data)
    return value, matrix


def random_unit_linf_direction(rng: np.random.Generator, dimension: int) -> np.ndarray:
    direction = rng.normal(size=dimension)
    scale = np.max(np.abs(direction))
    return direction / scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-refine-starts", type=int, default=24)
    parser.add_argument("--active-refine-scales", default="1e-4,1e-3,1e-2,5e-2")
    parser.add_argument("--active-refine-seed", type=int, default=20260530)
    parser.add_argument("--direction-count", type=int, default=12)
    parser.add_argument("--step", type=float, default=1e-5)
    parser.add_argument("--test-radii", default="1e-6,3e-6,1e-5,3e-5,1e-4")
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    start_time = time.time()
    _, support_modes, support_start = support_from_warm()
    support_problem = build_prob(support_modes)
    support_value, support_params, support_grad, refine_rows = refine_active_support(
        support_problem,
        support_start,
        args.active_refine_starts,
        parse_float_list(args.active_refine_scales),
        args.active_refine_seed,
    )
    full_problem = build_problem_scope("full-block")
    base_value, base_hessian = inactive_hessian_for_params(full_problem, support_modes, support_params)
    base_eigenvalues = np.linalg.eigvalsh(base_hessian)
    base_min = float(base_eigenvalues[0])
    components = connected_components(base_hessian, 1e-14)
    component_base = []
    for component in components:
        block = base_hessian[np.ix_(component, component)]
        eigenvalues = np.linalg.eigvalsh(block)
        component_base.append(
            {
                "dimension": len(component),
                "coordinates": [int(index) for index in component],
                "lambda_min_midpoint": float(eigenvalues[0]),
                "lambda_max_midpoint": float(eigenvalues[-1]),
            }
        )

    rng = np.random.default_rng(args.seed)
    directions = []
    # Coordinate probes for the least expensive exact-axis sensitivity sample.
    for index in range(min(len(support_params), args.direction_count)):
        direction = np.zeros_like(support_params)
        direction[index] = 1.0
        directions.append((f"coord:{index}", direction))
    while len(directions) < args.direction_count:
        directions.append((f"random:{len(directions)}", random_unit_linf_direction(rng, len(support_params))))

    derivative_rows = []
    coordinate_derivatives = []
    max_entry_derivative = 0.0
    max_operator_derivative = 0.0
    for label, direction in directions:
        plus_value, plus_hessian = inactive_hessian_for_params(full_problem, support_modes, support_params + args.step * direction)
        minus_value, minus_hessian = inactive_hessian_for_params(full_problem, support_modes, support_params - args.step * direction)
        derivative = (plus_hessian - minus_hessian) / (2.0 * args.step)
        entry_norm = float(np.max(np.abs(derivative)))
        operator_norm = float(np.linalg.norm(derivative, ord=2))
        max_entry_derivative = max(max_entry_derivative, entry_norm)
        max_operator_derivative = max(max_operator_derivative, operator_norm)
        if label.startswith("coord:"):
            coordinate_derivatives.append(derivative)
        derivative_rows.append(
            {
                "label": label,
                "max_entry_derivative": entry_norm,
                "operator_norm_derivative": operator_norm,
                "plus_value": plus_value,
                "minus_value": minus_value,
            }
        )
        print(f"{label:10s}: entry_deriv={entry_norm:.6e} op_deriv={operator_norm:.6e}", flush=True)

    radius_rows = []
    for radius in parse_float_list(args.test_radii):
        worst_lower = base_min - max_operator_derivative * radius
        entry_lower = base_min - base_hessian.shape[0] * max_entry_derivative * radius
        component_box_lowers = []
        for base_row, component in zip(component_base, components):
            if coordinate_derivatives:
                entry_l1 = 0.0
                operator_l1 = 0.0
                for derivative in coordinate_derivatives:
                    block_derivative = derivative[np.ix_(component, component)]
                    entry_l1 += float(np.max(np.abs(block_derivative)))
                    operator_l1 += float(np.linalg.norm(block_derivative, ord=2))
            else:
                entry_l1 = float("nan")
                operator_l1 = float("nan")
            component_box_lowers.append(
                {
                    "dimension": base_row["dimension"],
                    "lambda_min_midpoint": base_row["lambda_min_midpoint"],
                    "coordinate_entry_l1": entry_l1,
                    "coordinate_operator_l1": operator_l1,
                    "entrywise_box_lower": base_row["lambda_min_midpoint"] - base_row["dimension"] * entry_l1 * radius,
                    "operator_box_lower": base_row["lambda_min_midpoint"] - operator_l1 * radius,
                }
            )
        worst_component_entry_box = min(item["entrywise_box_lower"] for item in component_box_lowers)
        worst_component_operator_box = min(item["operator_box_lower"] for item in component_box_lowers)
        radius_rows.append(
            {
                "radius": radius,
                "operator_norm_linear_lower": worst_lower,
                "entrywise_full_dimension_lower": entry_lower,
                "worst_component_entrywise_coordinate_box_lower": worst_component_entry_box,
                "worst_component_operator_coordinate_box_lower": worst_component_operator_box,
                "component_coordinate_box_lowers": component_box_lowers,
            }
        )

    print("\nSummary")
    print("-------")
    print(f"base value={base_value:.17g} support value={support_value:.17g} grad={support_grad:.3e}")
    print(f"base inactive min eigenvalue={base_min:.12e}")
    print(f"max sampled operator derivative={max_operator_derivative:.6e}")
    print(f"max sampled entry derivative={max_entry_derivative:.6e}")
    for row in radius_rows:
        print(
            f"radius={row['radius']:.1e}: op-linear lower={row['operator_norm_linear_lower']:+.6e} "
            f"entry-full lower={row['entrywise_full_dimension_lower']:+.6e} "
            f"coord-box entry lower={row['worst_component_entrywise_coordinate_box_lower']:+.6e} "
            f"coord-box op lower={row['worst_component_operator_coordinate_box_lower']:+.6e}"
        )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "stability_probe_not_interval_certificate",
        "active_refine_starts_per_scale": args.active_refine_starts,
        "active_refine_top": sorted(refine_rows, key=lambda item: item["value"], reverse=True)[:20],
        "support_value": support_value,
        "support_grad": support_grad,
        "base_value": base_value,
        "base_inactive_min_eigenvalue": base_min,
        "base_inactive_max_eigenvalue": float(base_eigenvalues[-1]),
        "component_base": component_base,
        "step": args.step,
        "direction_count": len(directions),
        "max_sampled_entry_derivative": max_entry_derivative,
        "max_sampled_operator_derivative": max_operator_derivative,
        "derivative_rows": derivative_rows,
        "radius_rows": radius_rows,
        "elapsed_seconds": time.time() - start_time,
    }
    if not args.no_save:
        output = Path(args.output) if args.output else Path("results") / f"k3_inactive_hessian_stability_probe_{datetime.now():%Y%m%d_%H%M%S}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved: {output}")


if __name__ == "__main__":
    main()