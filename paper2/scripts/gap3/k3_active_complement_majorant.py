#!/usr/bin/env python3
"""Active/complement tensor majorant for the k=3 full-block problem.

This is a certificate scaffold for the global k=3 equality theorem.  Fix one
of the 12 active-orbit supports S and split a full field as u=a+b, where a is
supported on S and b is supported on the complement.  The active-only term is
bounded by the reduced active theorem.  The mixed and complement terms are
bounded by finite real-coordinate tensor flattening norms.

For the groups with exactly m active slots, the unweighted trilinear form has
zero polarized sum, so the output weight |ell|^2 may be replaced by
|ell|^2-center within each group.  This script scans centers, computes the
flattening upper bounds, and optimizes the resulting one-variable majorant in
the complement X-energy fraction q.

The result is not, by itself, the full theorem unless the majorant falls below
C3 on the region not already covered by the local bridge.  Its purpose is to
turn the global exclusion problem into explicit finite constants and identify
which active/complement group is too loose if the first majorant fails.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.gap3.k3_active_set_verify import build_problem_scope  # noqa: E402
from scripts.gap3.k3_fullblock_parallel_census import (  # noqa: E402
    build_orbit_support_indices,
    worker_count,
)


C3_TARGET = 0.021936469459403747249299192478957700397867315103825
SHELL_MIN = 8.0


@dataclass(frozen=True)
class CoordEntry:
    row: int
    vector: np.ndarray


def parse_centers(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def coordinate_tables(problem: dict, active_modes: set[int]) -> tuple[list[list[CoordEntry]], list[list[CoordEntry]], int, int]:
    n_modes = int(problem["N"])
    active_rows: dict[tuple[int, int, int], int] = {}
    complement_rows: dict[tuple[int, int, int], int] = {}
    for mode_index in range(n_modes):
        target = active_rows if mode_index in active_modes else complement_rows
        for polarization in range(2):
            for part in range(2):
                target[(mode_index, polarization, part)] = len(target)

    active_by_raw: list[list[CoordEntry]] = [[] for _ in range(2 * n_modes)]
    complement_by_raw: list[list[CoordEntry]] = [[] for _ in range(2 * n_modes)]

    for mode_index in range(n_modes):
        scale = math.sqrt(2.0 * float(problem["k2s"][mode_index]))
        for polarization in range(2):
            basis_vector = problem["e1s"][mode_index] if polarization == 0 else problem["e2s"][mode_index]
            for part, coefficient in enumerate((1.0 + 0.0j, 0.0 + 1.0j)):
                key = (mode_index, polarization, part)
                vector_pos = coefficient * basis_vector / scale
                vector_neg = np.conjugate(coefficient) * basis_vector / scale
                if mode_index in active_modes:
                    row = active_rows[key]
                    active_by_raw[mode_index].append(CoordEntry(row, vector_pos.astype(np.complex128)))
                    active_by_raw[mode_index + n_modes].append(CoordEntry(row, vector_neg.astype(np.complex128)))
                else:
                    row = complement_rows[key]
                    complement_by_raw[mode_index].append(CoordEntry(row, vector_pos.astype(np.complex128)))
                    complement_by_raw[mode_index + n_modes].append(CoordEntry(row, vector_neg.astype(np.complex128)))

    return active_by_raw, complement_by_raw, len(active_rows), len(complement_rows)


def tensor_shape(active_slots: int, active_dim: int, complement_dim: int) -> tuple[int, int, int]:
    if active_slots == 2:
        return active_dim, active_dim, complement_dim
    if active_slots == 1:
        return active_dim, complement_dim, complement_dim
    if active_slots == 0:
        return complement_dim, complement_dim, complement_dim
    raise ValueError("active_slots must be 0, 1, or 2")


def axis_indices(pattern: tuple[str, str, str], rows: tuple[int, int, int], active_slots: int) -> tuple[int, int, int]:
    active_rows: list[int] = []
    complement_rows: list[int] = []
    for space, row in zip(pattern, rows):
        if space == "A":
            active_rows.append(row)
        else:
            complement_rows.append(row)
    if active_slots == 2:
        return active_rows[0], active_rows[1], complement_rows[0]
    if active_slots == 1:
        return active_rows[0], complement_rows[0], complement_rows[1]
    return complement_rows[0], complement_rows[1], complement_rows[2]


def build_group_tensor(problem: dict, active_modes: set[int], active_slots: int, center: float) -> np.ndarray:
    active_by_raw, complement_by_raw, active_dim, complement_dim = coordinate_tables(problem, active_modes)
    tensor = np.zeros(tensor_shape(active_slots, active_dim, complement_dim), dtype=np.float64)
    patterns = [pattern for pattern in itertools.product(("A", "C"), repeat=3) if pattern.count("A") == active_slots]

    for ell_raw, ell2, r_raw, s_raw, s_vector in zip(
        problem["ell_idx"], problem["ell2"], problem["r_idx"], problem["s_idx"], problem["s_mat"]
    ):
        weight = float(ell2) - center
        slot_tables = {
            "A": (active_by_raw[int(ell_raw)], active_by_raw[int(r_raw)], active_by_raw[int(s_raw)]),
            "C": (complement_by_raw[int(ell_raw)], complement_by_raw[int(r_raw)], complement_by_raw[int(s_raw)]),
        }
        for pattern in patterns:
            ell_entries, r_entries, s_entries = (slot_tables[pattern[index]][index] for index in range(3))
            if not ell_entries or not r_entries or not s_entries:
                continue
            for ell_entry in ell_entries:
                ell_conj = np.conjugate(ell_entry.vector)
                for r_entry in r_entries:
                    r_linear = np.dot(s_vector, r_entry.vector)
                    if r_linear == 0.0:
                        continue
                    for s_entry in s_entries:
                        coeff = -float(np.imag(weight * r_linear * np.dot(ell_conj, s_entry.vector)))
                        if coeff == 0.0:
                            continue
                        tensor[axis_indices(pattern, (ell_entry.row, r_entry.row, s_entry.row), active_slots)] += coeff
    return tensor


def flattening_upper(tensor: np.ndarray) -> tuple[float, list[float], float]:
    dims = tensor.shape
    norms: list[float] = []
    for axis in range(3):
        matrix = np.moveaxis(tensor, axis, 0).reshape(dims[axis], -1)
        gram = matrix @ matrix.T
        top = float(np.linalg.eigvalsh(gram)[-1])
        norms.append(math.sqrt(max(top, 0.0)))
    return min(norms), norms, float(np.linalg.norm(tensor.ravel()))


def group_constants(problem: dict, support: list[int], centers: list[float]) -> list[dict[str, Any]]:
    active_modes = {int(index) for index in support}
    rows: list[dict[str, Any]] = []
    for active_slots in (2, 1, 0):
        best: dict[str, Any] | None = None
        for center in centers:
            tensor = build_group_tensor(problem, active_modes, active_slots, center)
            upper, flattening_norms, frobenius = flattening_upper(tensor)
            nnz = int(np.count_nonzero(np.abs(tensor) > 0.0))
            row = {
                "active_slots": active_slots,
                "center": center,
                "shape": list(tensor.shape),
                "nonzero_entries": nnz,
                "flattening_upper": upper,
                "flattening_norms": flattening_norms,
                "frobenius_norm": frobenius,
                "ratio_constant": upper / math.sqrt(SHELL_MIN),
            }
            if best is None or row["flattening_upper"] < best["flattening_upper"]:
                best = row
        assert best is not None
        rows.append(best)
    rows.sort(key=lambda item: -item["active_slots"])
    return rows


def majorant_value(q_value: float, constants: dict[int, float]) -> float:
    q = min(max(float(q_value), 0.0), 1.0)
    a = max(0.0, 1.0 - q)
    return (
        C3_TARGET * a * a
        + constants[2] * a * math.sqrt(q)
        + constants[1] * math.sqrt(a) * q
        + constants[0] * q ** 1.5
    )


def optimize_majorant(constants: dict[int, float]) -> dict[str, Any]:
    result = minimize_scalar(lambda q: -majorant_value(q, constants), bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-15})
    candidates = [0.0, 1.0, float(result.x)]
    best_q = max(candidates, key=lambda q: majorant_value(q, constants))
    best_value = majorant_value(best_q, constants)
    return {
        "max_value": best_value,
        "max_q": best_q,
        "gap_to_target": C3_TARGET - best_value,
        "endpoint_q0": majorant_value(0.0, constants),
        "endpoint_q1": majorant_value(1.0, constants),
    }


def run_support(task: tuple[int, list[int], list[float], dict]) -> dict[str, Any]:
    support_index, support, centers, problem = task
    rows = group_constants(problem, support, centers)
    constants = {int(row["active_slots"]): float(row["ratio_constant"]) for row in rows}
    return {
        "support_index": support_index,
        "support": [int(index) for index in support],
        "group_bounds": rows,
        "majorant": optimize_majorant(constants),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--centers", default="8,9,10,11,12,13,14")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--support-limit", type=int, default=0, help="limit orbit supports for quick probes; 0 uses all")
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    start_time = time.time()
    centers = parse_centers(args.centers)
    problem = build_problem_scope("full-block")
    supports = build_orbit_support_indices(problem)
    if args.support_limit:
        supports = supports[: args.support_limit]
    workers = min(worker_count(args.workers), max(1, len(supports)))
    tasks = [(index, support, centers, problem) for index, support in enumerate(supports)]

    print("k=3 active/complement tensor majorant")
    print("======================================")
    print(f"problem: N={problem['N']} triads={len(problem['ell_idx'])} orbit_supports={len(supports)} centers={centers}")
    print(f"workers={workers}", flush=True)

    results: list[dict[str, Any]] = []
    if workers == 1:
        for task in tasks:
            row = run_support(task)
            results.append(row)
            print(
                f"support {row['support_index']:2d}: majorant={row['majorant']['max_value']:.12g} "
                f"gap={row['majorant']['gap_to_target']:+.3e} q={row['majorant']['max_q']:.6g}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_support, task) for task in tasks]
            for future in as_completed(futures):
                row = future.result()
                results.append(row)
                print(
                    f"support {row['support_index']:2d}: majorant={row['majorant']['max_value']:.12g} "
                    f"gap={row['majorant']['gap_to_target']:+.3e} q={row['majorant']['max_q']:.6g}",
                    flush=True,
                )

    results.sort(key=lambda item: item["majorant"]["max_value"], reverse=True)
    worst = results[0]
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "active_complement_majorant",
        "target": C3_TARGET,
        "shell_min_for_denominator": SHELL_MIN,
        "centers": centers,
        "workers": workers,
        "problem_modes": int(problem["N"]),
        "problem_triads": int(len(problem["ell_idx"])),
        "orbit_support_count": len(supports),
        "worst_majorant": worst,
        "all_supports": results,
        "elapsed_seconds": time.time() - start_time,
        "method": "Active-only reduced theorem plus centered tensor flattening bounds for AAB/ABB/BBB groups; denominator bounded by sqrt(8).",
    }
    print("\nWorst majorant")
    print(f"  support={worst['support_index']} value={worst['majorant']['max_value']:.15f}")
    print(f"  target={C3_TARGET:.15f} gap={worst['majorant']['gap_to_target']:+.6e}")
    for row in worst["group_bounds"]:
        print(
            f"  active_slots={row['active_slots']} center={row['center']:g} "
            f"upper={row['flattening_upper']:.12g} ratio_const={row['ratio_constant']:.12g} shape={row['shape']}"
        )

    if not args.no_save:
        output = Path(args.output) if args.output else Path("results") / f"k3_active_complement_majorant_{datetime.now():%Y%m%d_%H%M%S}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved: {output}")


if __name__ == "__main__":
    main()