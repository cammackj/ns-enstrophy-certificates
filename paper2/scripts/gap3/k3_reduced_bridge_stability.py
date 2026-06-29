#!/usr/bin/env python3
"""Reduced-manifold inactive Hessian stability for the k=3 bridge.

This is the local bridge sizing script after the explicit reduced-to-active map:
it varies the three certified reduced variables (log_t, log_r, theta), embeds
the corresponding active coefficients into the full 85-mode block, and measures
the inactive Cartesian Hessian of -R.

The finite-difference rows are diagnostics, not interval derivative bounds.
They are meant to size the final interval proof over the reduced local box.
"""

from __future__ import annotations

import argparse
import itertools
import json
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

from scripts.gap3.k3_active_set_verify import build_problem_scope  # noqa: E402
from scripts.gap3.k3_closed_form_probe import support_from_warm  # noqa: E402
from scripts.gap3.k3_fullblock_cartesian_hessian import (  # noqa: E402
    coefficients_to_real,
    inactive_quadratic_data,
    inactive_quadratic_hessian,
    objective_from_real,
    real_to_coefficients,
)
from scripts.gap3.k3_reduced_active_map import active_coefficients_from_reduced  # noqa: E402


DEFAULT_CENTER = (-0.22384360556449978, -1.3921663408045954, 3.6136991131024327)


def parse_triple(text: str) -> tuple[float, float, float]:
    parts = [float(part.strip()) for part in text.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("expected three comma-separated values")
    return parts[0], parts[1], parts[2]


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


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


def embed_reduced_coefficients(
    full_problem: dict,
    support_modes: list[tuple[int, int, int]],
    reduced_point: tuple[float, float, float],
    scale: float,
) -> tuple[np.ndarray, list[int]]:
    full_index = {tuple(int(component) for component in wavevector): index for index, wavevector in enumerate(full_problem["wavevecs"])}
    active_coeffs = scale * active_coefficients_from_reduced(*reduced_point)
    full_coeffs = np.zeros(2 * full_problem["N"], dtype=np.complex128)
    active_indices = []
    for support_index, mode in enumerate(support_modes):
        index = full_index[tuple(int(component) for component in mode)]
        full_coeffs[2 * index : 2 * index + 2] = active_coeffs[2 * support_index : 2 * support_index + 2]
        active_indices.append(index)
    return coefficients_to_real(full_coeffs), active_indices


def inactive_hessian_for_reduced(
    full_problem: dict,
    support_modes: list[tuple[int, int, int]],
    reduced_point: tuple[float, float, float],
    scale: float,
) -> tuple[float, np.ndarray]:
    vector, active_indices = embed_reduced_coefficients(full_problem, support_modes, reduced_point, scale)
    value = objective_from_real(full_problem, vector)
    base_coefficients = real_to_coefficients(vector)
    inactive_data = inactive_quadratic_data(full_problem, active_indices)
    matrix = inactive_quadratic_hessian(full_problem, base_coefficients, value, inactive_data)
    return value, matrix


def component_base_rows(matrix: np.ndarray, components: list[list[int]]) -> list[dict]:
    rows = []
    for component in components:
        block = matrix[np.ix_(component, component)]
        eigenvalues = np.linalg.eigvalsh(block)
        rows.append(
            {
                "dimension": len(component),
                "coordinates": [int(index) for index in component],
                "lambda_min_midpoint": float(eigenvalues[0]),
                "lambda_max_midpoint": float(eigenvalues[-1]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--center", default=",".join(repr(item) for item in DEFAULT_CENTER))
    parser.add_argument("--radius", default="0.0002,0.0002,0.0002")
    parser.add_argument("--step", type=float, default=1e-5)
    parser.add_argument("--test-radii", default="1e-5,3e-5,1e-4,2e-4")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--zero-threshold", type=float, default=1e-14)
    parser.add_argument("--skip-corners", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    start_time = time.time()
    center = parse_triple(args.center)
    radius = parse_triple(args.radius)
    _, support_modes, _ = support_from_warm()
    full_problem = build_problem_scope("full-block")

    base_value, base_hessian = inactive_hessian_for_reduced(full_problem, support_modes, center, args.scale)
    components = connected_components(base_hessian, args.zero_threshold)
    base_rows = component_base_rows(base_hessian, components)
    base_rows_by_lower = sorted(base_rows, key=lambda row: row["lambda_min_midpoint"])
    base_eigenvalues = np.linalg.eigvalsh(base_hessian)

    print("k=3 reduced-manifold inactive Hessian stability")
    print("================================================")
    print(f"center={center} radius={radius} step={args.step:g} scale={args.scale:g}")
    print(f"base value={base_value:.17g}")
    print(
        f"inactive Hessian: dim={base_hessian.shape[0]} min={base_eigenvalues[0]:+.12e} "
        f"components={len(components)} sizes={dict(Counter(len(component) for component in components))}"
    )

    derivatives = []
    derivative_rows = []
    names = ["log_t", "log_r", "theta"]
    for axis, name in enumerate(names):
        direction = np.zeros(3, dtype=float)
        direction[axis] = 1.0
        plus = tuple(float(center[index] + args.step * direction[index]) for index in range(3))
        minus = tuple(float(center[index] - args.step * direction[index]) for index in range(3))
        plus_value, plus_hessian = inactive_hessian_for_reduced(full_problem, support_modes, plus, args.scale)
        minus_value, minus_hessian = inactive_hessian_for_reduced(full_problem, support_modes, minus, args.scale)
        derivative = (plus_hessian - minus_hessian) / (2.0 * args.step)
        derivatives.append(derivative)
        row = {
            "axis": name,
            "plus_value": plus_value,
            "minus_value": minus_value,
            "max_entry_derivative": float(np.max(np.abs(derivative))),
            "operator_norm_derivative": float(np.linalg.norm(derivative, ord=2)),
        }
        derivative_rows.append(row)
        print(
            f"{name:5s}: entry_deriv={row['max_entry_derivative']:.6e} "
            f"op_deriv={row['operator_norm_derivative']:.6e}"
        )

    radius_rows = []
    for test_radius in parse_float_list(args.test_radii):
        component_lowers = []
        for base_row, component in zip(base_rows, components):
            entry_l1 = 0.0
            operator_l1 = 0.0
            for derivative in derivatives:
                block = derivative[np.ix_(component, component)]
                entry_l1 += float(np.max(np.abs(block)))
                operator_l1 += float(np.linalg.norm(block, ord=2))
            component_lowers.append(
                {
                    "dimension": base_row["dimension"],
                    "lambda_min_midpoint": base_row["lambda_min_midpoint"],
                    "coordinate_entry_l1": entry_l1,
                    "coordinate_operator_l1": operator_l1,
                    "entrywise_box_lower": base_row["lambda_min_midpoint"] - base_row["dimension"] * entry_l1 * test_radius,
                    "operator_box_lower": base_row["lambda_min_midpoint"] - operator_l1 * test_radius,
                }
            )
        row = {
            "radius": test_radius,
            "worst_component_entrywise_coordinate_box_lower": min(item["entrywise_box_lower"] for item in component_lowers),
            "worst_component_operator_coordinate_box_lower": min(item["operator_box_lower"] for item in component_lowers),
            "component_coordinate_box_lowers": component_lowers,
        }
        radius_rows.append(row)
        print(
            f"radius={test_radius:.1e}: coord-box entry lower={row['worst_component_entrywise_coordinate_box_lower']:+.6e} "
            f"coord-box op lower={row['worst_component_operator_coordinate_box_lower']:+.6e}"
        )

    corner_rows = []
    if not args.skip_corners:
        print("corner samples:")
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            point = tuple(center[index] + signs[index] * radius[index] for index in range(3))
            value, matrix = inactive_hessian_for_reduced(full_problem, support_modes, point, args.scale)
            eigen_min = float(np.linalg.eigvalsh(matrix)[0])
            corner_rows.append({"signs": [int(sign) for sign in signs], "point": list(point), "value": value, "min_eigenvalue": eigen_min})
            print(f"  signs={signs}: value={value:.17g} min_eig={eigen_min:+.12e}")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "reduced_manifold_stability_probe_not_interval_certificate",
        "center": list(center),
        "radius": list(radius),
        "step": args.step,
        "scale": args.scale,
        "base_value": base_value,
        "base_inactive_min_eigenvalue": float(base_eigenvalues[0]),
        "base_inactive_max_eigenvalue": float(base_eigenvalues[-1]),
        "component_count": len(components),
        "component_size_counts": dict(Counter(len(component) for component in components)),
        "base_components_by_min_eigenvalue": base_rows_by_lower,
        "derivative_rows": derivative_rows,
        "radius_rows": radius_rows,
        "corner_rows": corner_rows,
        "elapsed_seconds": time.time() - start_time,
    }
    if not args.no_save:
        output = Path(args.output) if args.output else Path("results") / f"k3_reduced_bridge_stability_{datetime.now():%Y%m%d_%H%M%S}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved: {output}")


if __name__ == "__main__":
    main()