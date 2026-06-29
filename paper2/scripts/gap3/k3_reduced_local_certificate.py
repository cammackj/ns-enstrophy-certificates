#!/usr/bin/env python3
"""Local interval certificate for the reduced k=3 stationary point.

The script checks two local facts on a small box around the high-precision
stationary point:

1. Krawczyk uniqueness for the zero of grad(f).
2. Negative definiteness of the interval Hessian, using Gershgorin when it is
   available and a symmetric-vertex check as a fallback diagnostic.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import mpmath as mp
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.gap3.k3_reduced_interval_branch import Box, interval_lower, interval_upper  # noqa: E402
from scripts.gap3.k3_reduced_taylor_branch import point_box, reduced_ad  # noqa: E402
from scripts.gap3.k3_three_variable_reduction import k3_reduced  # noqa: E402


DEFAULT_CENTER = (
    -0.22384360556449978,
    -1.3921663408045954,
    3.6136991131024327,
)


def parse_triple(text: str) -> tuple[float, float, float]:
    parts = [float(part.strip()) for part in text.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("expected three comma-separated values")
    return parts[0], parts[1], parts[2]


def make_box(center: tuple[float, float, float], radius: tuple[float, float, float]) -> Box:
    return Box(
        (center[0] - radius[0], center[0] + radius[0]),
        (center[1] - radius[1], center[1] + radius[1]),
        (center[2] - radius[2], center[2] + radius[2]),
    )


def interval_mid(value: mp.iv.mpf) -> float:
    return 0.5 * (interval_lower(value) + interval_upper(value))


def interval_abs_upper(value: mp.iv.mpf) -> float:
    return max(abs(interval_lower(value)), abs(interval_upper(value)))


def point_interval(value: float) -> mp.iv.mpf:
    text = repr(float(value))
    return mp.iv.mpf([text, text])


def interval_add(left: mp.iv.mpf, right: mp.iv.mpf) -> mp.iv.mpf:
    return left + right


def interval_matrix_vector(matrix: list[list[mp.iv.mpf]], vector: list[mp.iv.mpf]) -> list[mp.iv.mpf]:
    rows = []
    for row in matrix:
        total = mp.iv.mpf([0, 0])
        for entry, component in zip(row, vector):
            total += entry * component
        rows.append(total)
    return rows


def numeric_left_multiply(matrix: np.ndarray, vector: list[mp.iv.mpf]) -> list[mp.iv.mpf]:
    rows = []
    for i in range(matrix.shape[0]):
        total = mp.iv.mpf([0, 0])
        for j in range(matrix.shape[1]):
            total += point_interval(float(matrix[i, j])) * vector[j]
        rows.append(total)
    return rows


def numeric_interval_matrix_product(matrix: np.ndarray, interval_matrix: list[list[mp.iv.mpf]]) -> list[list[mp.iv.mpf]]:
    rows = []
    for i in range(matrix.shape[0]):
        row = []
        for j in range(len(interval_matrix[0])):
            total = mp.iv.mpf([0, 0])
            for k in range(matrix.shape[1]):
                total += point_interval(float(matrix[i, k])) * interval_matrix[k][j]
            row.append(total)
        rows.append(row)
    return rows


def interval_hessian(box: Box) -> list[list[mp.iv.mpf]]:
    ad_value = reduced_ad(box, track_hessian=True)
    if ad_value.hess is None:
        raise RuntimeError("Hessian tracking did not produce a Hessian")
    return [[ad_value.hess[i][j] for j in range(3)] for i in range(3)]


def midpoint_matrix(hessian: list[list[mp.iv.mpf]]) -> np.ndarray:
    return np.array([[interval_mid(entry) for entry in row] for row in hessian], dtype=float)


def krawczyk_check(center: tuple[float, float, float], radius: tuple[float, float, float], hessian: list[list[mp.iv.mpf]]) -> dict:
    center_ad = reduced_ad(point_box(center), track_hessian=False)
    gradient = list(center_ad.grad)
    point_hessian = interval_hessian(point_box(center))
    inverse_mid = np.linalg.inv(midpoint_matrix(point_hessian))

    y_gradient = numeric_left_multiply(inverse_mid, gradient)
    y_hessian = numeric_interval_matrix_product(inverse_mid, hessian)
    identity_minus_yh = []
    for i in range(3):
        row = []
        for j in range(3):
            identity = mp.iv.mpf([1, 1]) if i == j else mp.iv.mpf([0, 0])
            row.append(identity - y_hessian[i][j])
        identity_minus_yh.append(row)

    delta = [mp.iv.mpf([-item, item]) for item in radius]
    remainder = interval_matrix_vector(identity_minus_yh, delta)
    krawczyk_relative = [(-yg) + rem for yg, rem in zip(y_gradient, remainder)]
    subset = [interval_lower(item) > -rad and interval_upper(item) < rad for item, rad in zip(krawczyk_relative, radius)]
    return {
        "gradient_abs_upper": [interval_abs_upper(item) for item in gradient],
        "inverse_mid_hessian": inverse_mid.tolist(),
        "krawczyk_relative": [[interval_lower(item), interval_upper(item)] for item in krawczyk_relative],
        "subset_flags": subset,
        "passes": all(subset),
    }


def symmetrized_hessian_bounds(hessian: list[list[mp.iv.mpf]]) -> list[list[tuple[float, float]]]:
    bounds = [[(0.0, 0.0) for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            lo = min(interval_lower(hessian[i][j]), interval_lower(hessian[j][i]))
            hi = max(interval_upper(hessian[i][j]), interval_upper(hessian[j][i]))
            bounds[i][j] = (lo, hi)
    return bounds


def negative_definite_checks(hessian: list[list[mp.iv.mpf]]) -> dict:
    bounds = symmetrized_hessian_bounds(hessian)
    gershgorin_rows = []
    for i in range(3):
        row_upper = bounds[i][i][1]
        radius = sum(max(abs(bounds[i][j][0]), abs(bounds[i][j][1])) for j in range(3) if j != i)
        gershgorin_rows.append(row_upper + radius)
    gershgorin_passes = all(row < 0 for row in gershgorin_rows)

    entries = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    max_vertex_eigenvalue = -math.inf
    worst_vertex = None
    for choices in itertools.product([0, 1], repeat=len(entries)):
        matrix = np.zeros((3, 3), dtype=float)
        for bit, (i, j) in zip(choices, entries):
            value = bounds[i][j][bit]
            matrix[i, j] = value
            matrix[j, i] = value
        eigenvalues = np.linalg.eigvalsh(matrix)
        top = float(eigenvalues[-1])
        if top > max_vertex_eigenvalue:
            max_vertex_eigenvalue = top
            worst_vertex = matrix.tolist()
    return {
        "hessian_bounds": [[[lo, hi] for lo, hi in row] for row in bounds],
        "gershgorin_row_upper_bounds": gershgorin_rows,
        "gershgorin_passes": gershgorin_passes,
        "max_symmetric_vertex_eigenvalue": max_vertex_eigenvalue,
        "symmetric_vertex_passes": max_vertex_eigenvalue < 0,
        "worst_vertex": worst_vertex,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--center", default=",".join(repr(item) for item in DEFAULT_CENTER))
    parser.add_argument("--radius", default="0.0002,0.0002,0.0002")
    parser.add_argument("--dps", type=int, default=80)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    mp.iv.dps = args.dps
    center = parse_triple(args.center)
    radius = parse_triple(args.radius)
    box = make_box(center, radius)
    hessian = interval_hessian(box)
    krawczyk = krawczyk_check(center, radius, hessian)
    negativity = negative_definite_checks(hessian)
    value_at_center = k3_reduced(math.exp(center[0]), math.exp(center[1]), center[2])

    print("k=3 reduced local interval certificate")
    print("=======================================")
    print(f"center={center}")
    print(f"radius={radius}")
    print(f"value_at_center={value_at_center:.17g}")
    print(f"gradient abs upper={krawczyk['gradient_abs_upper']}")
    print(f"Krawczyk passes={krawczyk['passes']} relative={krawczyk['krawczyk_relative']}")
    print(f"Gershgorin passes={negativity['gershgorin_passes']} rows={negativity['gershgorin_row_upper_bounds']}")
    print(
        "symmetric vertex passes="
        f"{negativity['symmetric_vertex_passes']} max_eigen={negativity['max_symmetric_vertex_eigenvalue']:.17g}"
    )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "center": list(center),
        "radius": list(radius),
        "box": box.as_list(),
        "value_at_center": value_at_center,
        "krawczyk": krawczyk,
        "negative_definite": negativity,
        "passes": bool(krawczyk["passes"] and (negativity["gershgorin_passes"] or negativity["symmetric_vertex_passes"])),
    }
    if not args.no_save:
        output = Path(args.output) if args.output else Path("results") / f"k3_reduced_local_certificate_{datetime.now():%Y%m%d_%H%M%S}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved: {output}")


if __name__ == "__main__":
    main()