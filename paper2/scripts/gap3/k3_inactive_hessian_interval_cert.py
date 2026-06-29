#!/usr/bin/env python3
"""Interval-style certificate for the k=3 inactive Hessian blocks.

Given a bridge ledger containing the sparse inactive Hessian, this script checks
positive definiteness component-by-component under a uniform entrywise error
budget.  If every entry is known within +/- eps, Weyl's inequality gives

    lambda_min(A + E) >= lambda_min(A) - n eps

for each n-by-n component, since ||E||_2 <= ||E||_F <= n eps.

This is deliberately simple and auditable.  It converts the floating bridge
matrix into a finite margin ledger and tells us how much interval/rational
rounding error the full-block local bridge can tolerate.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np


def load_sparse_matrix(data: dict) -> np.ndarray:
    dimension = int(data["inactive_hessian_dimension"])
    matrix = np.zeros((dimension, dimension), dtype=float)
    for entry in data.get("inactive_hessian_sparse_entries", []):
        row = int(entry["row"])
        column = int(entry["column"])
        value = float(entry["value"])
        matrix[row, column] = value
        matrix[column, row] = value
    return matrix


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


def component_certificate(matrix: np.ndarray, component: list[int], entry_radius: float) -> dict:
    block = matrix[np.ix_(component, component)]
    eigenvalues = np.linalg.eigvalsh(block)
    dimension = len(component)
    frobenius_error_bound = dimension * entry_radius
    certified_lower = float(eigenvalues[0] - frobenius_error_bound)
    diagonal = np.diag(block)
    off_diagonal = np.sum(np.abs(block), axis=1) - np.abs(diagonal)
    gershgorin_margin = float(np.min(diagonal - off_diagonal))
    return {
        "dimension": int(dimension),
        "coordinates": [int(index) for index in component],
        "lambda_min_midpoint": float(eigenvalues[0]),
        "lambda_max_midpoint": float(eigenvalues[-1]),
        "entry_radius": entry_radius,
        "frobenius_error_bound": frobenius_error_bound,
        "certified_lambda_lower": certified_lower,
        "passes": certified_lower > 0.0,
        "min_gershgorin_margin_midpoint": gershgorin_margin,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-json", default="results/k3_fullblock_bridge_certificate_refined_20260531.json")
    parser.add_argument("--entry-radius", type=float, default=1e-10)
    parser.add_argument("--zero-threshold", type=float, default=1e-14)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    data = json.loads(Path(args.bridge_json).read_text(encoding="utf-8"))
    matrix = load_sparse_matrix(data)
    if not np.allclose(matrix, matrix.T, atol=0.0, rtol=0.0):
        raise RuntimeError("inactive Hessian sparse entries did not reconstruct a symmetric matrix")
    components = connected_components(matrix, args.zero_threshold)
    rows = [component_certificate(matrix, component, args.entry_radius) for component in components]
    rows.sort(key=lambda item: item["certified_lambda_lower"])
    pass_all = all(row["passes"] for row in rows)
    max_allowed_entry_radius = min(row["lambda_min_midpoint"] / row["dimension"] for row in rows)

    print("k=3 inactive Hessian interval-style certificate")
    print("================================================")
    print(f"bridge: {args.bridge_json}")
    print(f"dimension={matrix.shape[0]} components={len(rows)} size_counts={dict(Counter(row['dimension'] for row in rows))}")
    print(f"entry_radius={args.entry_radius:.3e} pass={pass_all}")
    print(f"worst certified lower={rows[0]['certified_lambda_lower']:+.12e}")
    print(f"max allowed uniform entry radius={max_allowed_entry_radius:.12e}")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "bridge_json": args.bridge_json,
        "entry_radius": args.entry_radius,
        "zero_threshold": args.zero_threshold,
        "matrix_dimension": int(matrix.shape[0]),
        "component_count": int(len(rows)),
        "component_size_counts": dict(Counter(row["dimension"] for row in rows)),
        "passes": pass_all,
        "worst_certified_lambda_lower": rows[0]["certified_lambda_lower"],
        "worst_component": rows[0],
        "max_allowed_uniform_entry_radius": max_allowed_entry_radius,
        "components_by_certified_lower": rows,
        "method": "For each component, lambda_min(A+E) >= lambda_min(A) - n*entry_radius using ||E||_2 <= ||E||_F <= n*entry_radius.",
    }
    if not math.isfinite(summary["max_allowed_uniform_entry_radius"]):
        raise RuntimeError("non-finite interval margin")
    if not args.no_save:
        output = Path(args.output) if args.output else Path("results") / f"k3_inactive_hessian_interval_cert_{datetime.now():%Y%m%d_%H%M%S}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved: {output}")


if __name__ == "__main__":
    main()