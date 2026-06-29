#!/usr/bin/env python3
"""Interval inactive-Hessian certificate over the reduced k=3 local box."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import mpmath as mp
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.gap3.k3_active_set_verify import build_problem_scope  # noqa: E402
from scripts.gap3.k3_closed_form_probe import support_from_warm  # noqa: E402
from scripts.gap3.k3_reduced_active_map_interval import (  # noqa: E402
    IComplex,
    active_coefficients_interval,
    make_box,
    parse_triple,
)
from scripts.gap3.k3_reduced_interval_branch import interval_lower, interval_upper, reduced_iv, sqrt_nonnegative  # noqa: E402


DEFAULT_CENTER = (-0.22384360556449978, -1.3921663408045954, 3.6136991131024327)
COORDINATE_NAMES = ("e1_re", "e1_im", "e2_re", "e2_im")

_WORKER_PROBLEM = None
_WORKER_ACTIVE_U = None
_WORKER_COORDINATE_U = None
_WORKER_RECORDS_BY_PAIR = None
_WORKER_DENOMINATOR = None


def iv(value: float | int | str) -> mp.iv.mpf:
    text = str(value)
    return mp.iv.mpf([text, text])


def zero_c() -> IComplex:
    return IComplex(iv(0), iv(0))


def const_c(re_value: float, im_value: float = 0.0) -> IComplex:
    return IComplex(iv(re_value), iv(im_value))


def interval_mid(value: mp.iv.mpf) -> float:
    return 0.5 * (interval_lower(value) + interval_upper(value))


def interval_radius(value: mp.iv.mpf) -> float:
    return 0.5 * (interval_upper(value) - interval_lower(value))


def interval_payload(value: mp.iv.mpf) -> tuple[str, str]:
    return mp.nstr(mp.mpf(value.a), n=80), mp.nstr(mp.mpf(value.b), n=80)


def interval_from_payload(payload: tuple[str, str]) -> mp.iv.mpf:
    return mp.iv.mpf([payload[0], payload[1]])


def ic_payload(value: IComplex) -> tuple[tuple[str, str], tuple[str, str]]:
    return interval_payload(value.re), interval_payload(value.im)


def ic_from_payload(payload: tuple[tuple[str, str], tuple[str, str]]) -> IComplex:
    return IComplex(interval_from_payload(payload[0]), interval_from_payload(payload[1]))


def active_u_payload(active_u: dict[int, list[IComplex]]) -> dict[int, list[tuple[tuple[str, str], tuple[str, str]]]]:
    return {int(raw_index): [ic_payload(value) for value in vector] for raw_index, vector in active_u.items()}


def active_u_from_payload(payload: dict[int, list[tuple[tuple[str, str], tuple[str, str]]]]) -> dict[int, list[IComplex]]:
    return {int(raw_index): [ic_from_payload(value) for value in vector] for raw_index, vector in payload.items()}


def c_add(left: IComplex, right: IComplex) -> IComplex:
    return IComplex(left.re + right.re, left.im + right.im)


def c_mul(left: IComplex, right: IComplex) -> IComplex:
    return IComplex(left.re * right.re - left.im * right.im, left.re * right.im + left.im * right.re)


def c_scale(value: IComplex, factor: float | mp.iv.mpf) -> IComplex:
    return IComplex(value.re * factor, value.im * factor)


def c_conj(value: IComplex) -> IComplex:
    return IComplex(value.re, -value.im)


def c_dot_real(weights: np.ndarray, values: list[IComplex]) -> IComplex:
    total = zero_c()
    for weight, value in zip(weights, values):
        if float(weight) != 0.0:
            total = c_add(total, c_scale(value, float(weight)))
    return total


def make_full_coefficients(
    problem: dict,
    support_modes: list[tuple[int, int, int]],
    active_coeffs: list[IComplex],
    inactive_indices: list[int],
    assignments: dict[int, complex],
) -> list[list[IComplex]]:
    full_index = {tuple(int(component) for component in wavevector): index for index, wavevector in enumerate(problem["wavevecs"])}
    coeffs = [[zero_c(), zero_c()] for _ in range(problem["N"])]
    for support_index, mode in enumerate(support_modes):
        index = full_index[tuple(int(component) for component in mode)]
        coeffs[index] = [active_coeffs[2 * support_index], active_coeffs[2 * support_index + 1]]
    for coordinate, value in assignments.items():
        inactive_position = coordinate // 4
        mode_index = inactive_indices[inactive_position]
        local_coordinate = coordinate % 4
        if local_coordinate == 0:
            coeffs[mode_index][0] = c_add(coeffs[mode_index][0], const_c(value.real, value.imag))
        elif local_coordinate == 1:
            coeffs[mode_index][0] = c_add(coeffs[mode_index][0], const_c(-value.imag, value.real))
        elif local_coordinate == 2:
            coeffs[mode_index][1] = c_add(coeffs[mode_index][1], const_c(value.real, value.imag))
        elif local_coordinate == 3:
            coeffs[mode_index][1] = c_add(coeffs[mode_index][1], const_c(-value.imag, value.real))
        else:
            raise AssertionError(local_coordinate)
    return coeffs


def u_raw(problem: dict, coeffs: list[list[IComplex]], raw_index: int) -> list[IComplex]:
    n_modes = problem["N"]
    mode_index = raw_index % n_modes
    vector = []
    for component in range(3):
        value = c_add(
            c_scale(coeffs[mode_index][0], float(problem["e1s"][mode_index][component])),
            c_scale(coeffs[mode_index][1], float(problem["e2s"][mode_index][component])),
        )
        vector.append(c_conj(value) if raw_index >= n_modes else value)
    return vector


def b_value_interval(problem: dict, coeffs: list[list[IComplex]], triad_indices: np.ndarray) -> mp.iv.mpf:
    total = zero_c()
    for triad_index in triad_indices:
        ell_raw = int(problem["ell_idx"][triad_index])
        r_raw = int(problem["r_idx"][triad_index])
        s_raw = int(problem["s_idx"][triad_index])
        u_ell = u_raw(problem, coeffs, ell_raw)
        u_r = u_raw(problem, coeffs, r_raw)
        u_s = u_raw(problem, coeffs, s_raw)
        s_dot_ur = c_dot_real(problem["s_mat"][triad_index], u_r)
        conj_ell = [c_conj(value) for value in u_ell]
        ell_dot_s = zero_c()
        for left, right in zip(conj_ell, u_s):
            ell_dot_s = c_add(ell_dot_s, c_mul(left, right))
        term = c_scale(c_mul(s_dot_ur, ell_dot_s), float(problem["ell2"][triad_index]))
        total = c_add(total, term)
    return -total.im


def active_denominator_intervals(problem: dict, support_modes: list[tuple[int, int, int]], active_coeffs: list[IComplex], box) -> tuple[mp.iv.mpf, mp.iv.mpf, mp.iv.mpf, mp.iv.mpf]:
    x2 = iv(0)
    d2 = iv(0)
    for support_index, mode in enumerate(support_modes):
        shell = sum(component * component for component in mode)
        c1 = active_coeffs[2 * support_index]
        c2 = active_coeffs[2 * support_index + 1]
        amp = c1.re * c1.re + c1.im * c1.im + c2.re * c2.re + c2.im * c2.im
        x2 += 2 * shell * amp
        d2 += 2 * shell * shell * amp
    denominator = x2 * sqrt_nonnegative(d2)
    active_value = reduced_iv(box)
    return x2, d2, denominator, active_value


def inactive_indices_and_triads(problem: dict, active_indices: list[int]) -> tuple[list[int], np.ndarray]:
    active_set = set(active_indices)
    inactive_indices = [index for index in range(problem["N"]) if index not in active_set]
    triad_indices = []
    for triad_index in range(len(problem["ell_idx"])):
        bases = [
            int(problem["ell_idx"][triad_index]) % problem["N"],
            int(problem["r_idx"][triad_index]) % problem["N"],
            int(problem["s_idx"][triad_index]) % problem["N"],
        ]
        if sum(base not in active_set for base in bases) == 2:
            triad_indices.append(triad_index)
    return inactive_indices, np.array(triad_indices, dtype=np.int64)


def structural_pairs(problem: dict, active_indices: list[int], inactive_indices: list[int], triad_indices: np.ndarray) -> set[tuple[int, int]]:
    active_set = set(active_indices)
    inactive_position = {mode_index: position for position, mode_index in enumerate(inactive_indices)}
    pairs: set[tuple[int, int]] = set()
    for triad_index in triad_indices:
        bases = [
            int(problem["ell_idx"][triad_index]) % problem["N"],
            int(problem["r_idx"][triad_index]) % problem["N"],
            int(problem["s_idx"][triad_index]) % problem["N"],
        ]
        inactive_bases = [base for base in bases if base not in active_set]
        if len(inactive_bases) != 2 or inactive_bases[0] == inactive_bases[1]:
            continue
        left = inactive_position[inactive_bases[0]]
        right = inactive_position[inactive_bases[1]]
        for left_coordinate in range(4 * left, 4 * left + 4):
            for right_coordinate in range(4 * right, 4 * right + 4):
                i, j = sorted((left_coordinate, right_coordinate))
                pairs.add((i, j))
    return pairs


def active_u_raw_table(problem: dict, support_modes: list[tuple[int, int, int]], active_coeffs: list[IComplex]) -> dict[int, list[IComplex]]:
    full_coeffs = [[zero_c(), zero_c()] for _ in range(problem["N"])]
    full_index = {tuple(int(component) for component in wavevector): index for index, wavevector in enumerate(problem["wavevecs"])}
    active_raw = set()
    for support_index, mode in enumerate(support_modes):
        index = full_index[tuple(int(component) for component in mode)]
        full_coeffs[index] = [active_coeffs[2 * support_index], active_coeffs[2 * support_index + 1]]
        active_raw.add(index)
        active_raw.add(index + problem["N"])
    return {raw_index: u_raw(problem, full_coeffs, raw_index) for raw_index in sorted(active_raw)}


def coordinate_coefficients(local_coordinate: int) -> tuple[IComplex, IComplex]:
    if local_coordinate == 0:
        return const_c(1.0), zero_c()
    if local_coordinate == 1:
        return const_c(0.0, 1.0), zero_c()
    if local_coordinate == 2:
        return zero_c(), const_c(1.0)
    if local_coordinate == 3:
        return zero_c(), const_c(0.0, 1.0)
    raise AssertionError(local_coordinate)


def coordinate_u_raw_table(problem: dict, inactive_indices: list[int]) -> dict[tuple[int, int], list[IComplex]]:
    table = {}
    for coordinate in range(4 * len(inactive_indices)):
        inactive_position = coordinate // 4
        mode_index = inactive_indices[inactive_position]
        c1, c2 = coordinate_coefficients(coordinate % 4)
        vector = []
        for component in range(3):
            value = c_add(
                c_scale(c1, float(problem["e1s"][mode_index][component])),
                c_scale(c2, float(problem["e2s"][mode_index][component])),
            )
            vector.append(value)
        table[(coordinate, mode_index)] = vector
        table[(coordinate, mode_index + problem["N"])] = [c_conj(value) for value in vector]
    return table


def pair_records(
    problem: dict,
    active_indices: list[int],
    inactive_indices: list[int],
    triad_indices: np.ndarray,
) -> dict[tuple[int, int], list[tuple[int, tuple[int, int, int]]]]:
    active_set = set(active_indices)
    inactive_position = {mode_index: position for position, mode_index in enumerate(inactive_indices)}
    records: dict[tuple[int, int], list[tuple[int, tuple[int, int, int]]]] = defaultdict(list)
    for triad_index in triad_indices:
        raw_indices = (
            int(problem["ell_idx"][triad_index]),
            int(problem["r_idx"][triad_index]),
            int(problem["s_idx"][triad_index]),
        )
        bases = [raw_index % problem["N"] for raw_index in raw_indices]
        inactive_slots = [slot for slot, base in enumerate(bases) if base not in active_set]
        if len(inactive_slots) != 2:
            continue
        left_slot, right_slot = inactive_slots
        left_base = bases[left_slot]
        right_base = bases[right_slot]
        if left_base == right_base:
            continue
        left_position = inactive_position[left_base]
        right_position = inactive_position[right_base]
        for left_coordinate in range(4 * left_position, 4 * left_position + 4):
            for right_coordinate in range(4 * right_position, 4 * right_position + 4):
                slot_coordinates = [-1, -1, -1]
                slot_coordinates[left_slot] = left_coordinate
                slot_coordinates[right_slot] = right_coordinate
                i, j = sorted((left_coordinate, right_coordinate))
                records[(i, j)].append((int(triad_index), tuple(slot_coordinates)))
    return records


def b_pair_interval_fast(
    problem: dict,
    active_u: dict[int, list[IComplex]],
    coordinate_u: dict[tuple[int, int], list[IComplex]],
    records: list[tuple[int, tuple[int, int, int]]],
) -> mp.iv.mpf:
    total = zero_c()
    for triad_index, slot_coordinates in records:
        raw_indices = (
            int(problem["ell_idx"][triad_index]),
            int(problem["r_idx"][triad_index]),
            int(problem["s_idx"][triad_index]),
        )
        values = []
        for raw_index, coordinate in zip(raw_indices, slot_coordinates):
            if coordinate < 0:
                values.append(active_u[raw_index])
            else:
                values.append(coordinate_u[(coordinate, raw_index)])
        u_ell, u_r, u_s = values
        s_dot_ur = c_dot_real(problem["s_mat"][triad_index], u_r)
        conj_ell = [c_conj(value) for value in u_ell]
        ell_dot_s = zero_c()
        for left, right in zip(conj_ell, u_s):
            ell_dot_s = c_add(ell_dot_s, c_mul(left, right))
        total = c_add(total, c_scale(c_mul(s_dot_ur, ell_dot_s), float(problem["ell2"][triad_index])))
    return -total.im


def init_offdiag_worker(problem, active_u_data, inactive_indices, records_by_pair, denominator_data, dps: int) -> None:
    global _WORKER_PROBLEM, _WORKER_ACTIVE_U, _WORKER_COORDINATE_U, _WORKER_RECORDS_BY_PAIR, _WORKER_DENOMINATOR
    mp.iv.dps = dps
    _WORKER_PROBLEM = problem
    _WORKER_ACTIVE_U = active_u_from_payload(active_u_data)
    _WORKER_COORDINATE_U = coordinate_u_raw_table(problem, inactive_indices)
    _WORKER_RECORDS_BY_PAIR = records_by_pair
    _WORKER_DENOMINATOR = interval_from_payload(denominator_data)


def offdiag_worker(task: tuple[int, int, int]) -> tuple[int, int, int, float, float, float, float]:
    ordinal, i, j = task
    b_interval = b_pair_interval_fast(
        _WORKER_PROBLEM,
        _WORKER_ACTIVE_U,
        _WORKER_COORDINATE_U,
        _WORKER_RECORDS_BY_PAIR[(i, j)],
    )
    h_interval = -b_interval / _WORKER_DENOMINATOR
    lower = interval_lower(h_interval)
    upper = interval_upper(h_interval)
    return ordinal, i, j, lower, upper, 0.5 * (lower + upper), 0.5 * (upper - lower)


def resolve_workers(requested: int) -> int:
    if requested == 0:
        return max(1, os.cpu_count() or 1)
    return max(1, requested)


def connected_components_from_pairs(dimension: int, pairs: set[tuple[int, int]]) -> list[list[int]]:
    adjacency = [set() for _ in range(dimension)]
    for i, j in pairs:
        if i != j:
            adjacency[i].add(j)
            adjacency[j].add(i)
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


def coordinate_assignment(coordinate: int) -> dict[int, complex]:
    local_coordinate = coordinate % 4
    value = 1.0 if local_coordinate in (0, 2) else 1.0j
    return {coordinate: value}


def pair_assignment(i: int, j: int) -> dict[int, complex]:
    assignments = coordinate_assignment(i)
    assignments.update(coordinate_assignment(j))
    return assignments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--center", default=",".join(repr(item) for item in DEFAULT_CENTER))
    parser.add_argument("--radius", default="0.0002,0.0002,0.0002")
    parser.add_argument("--dps", type=int, default=50)
    parser.add_argument("--workers", type=int, default=0, help="parallel workers for off-diagonal entries; 0 uses all logical CPUs")
    parser.add_argument("--chunksize", type=int, default=8, help="task chunksize for multiprocessing")
    parser.add_argument("--progress-every", type=int, default=200, help="print progress every N off-diagonal entries; 0 disables")
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    start_time = time.time()
    mp.iv.dps = args.dps
    center = parse_triple(args.center)
    radius = parse_triple(args.radius)
    box = make_box(center, radius)
    active_coeffs, map_diagnostics = active_coefficients_interval(box)
    _, support_modes, _ = support_from_warm()
    problem = build_problem_scope("full-block")
    full_index = {tuple(int(component) for component in wavevector): index for index, wavevector in enumerate(problem["wavevecs"])}
    active_indices = [full_index[tuple(mode)] for mode in support_modes]
    inactive_indices, triad_indices = inactive_indices_and_triads(problem, active_indices)
    dimension = 4 * len(inactive_indices)
    records_by_pair = pair_records(problem, active_indices, inactive_indices, triad_indices)
    pairs = set(records_by_pair)
    components = connected_components_from_pairs(dimension, pairs)
    x2, d2, denominator, active_value = active_denominator_intervals(problem, support_modes, active_coeffs, box)

    print("k=3 reduced bridge interval inactive Hessian certificate")
    print("=========================================================")
    print(f"center={center} radius={radius} dps={args.dps}")
    print(f"inactive dimension={dimension} structural pairs={len(pairs)} components={len(components)} sizes={dict(Counter(len(c) for c in components))}")
    print(f"active value interval=[{interval_lower(active_value):.17g}, {interval_upper(active_value):.17g}]")

    mid = np.zeros((dimension, dimension), dtype=float)
    rad = np.zeros((dimension, dimension), dtype=float)
    diagonal_rows = []
    for coordinate in range(dimension):
        inactive_mode = inactive_indices[coordinate // 4]
        shell = float(problem["k2s"][inactive_mode])
        q_value = active_value * (2 * shell / x2 + shell * shell / d2)
        h_value = 2 * q_value
        mid[coordinate, coordinate] = interval_mid(h_value)
        rad[coordinate, coordinate] = interval_radius(h_value)
        diagonal_rows.append(
            {
                "coordinate": coordinate,
                "mode_index": inactive_mode,
                "coordinate_name": COORDINATE_NAMES[coordinate % 4],
                "interval": [interval_lower(h_value), interval_upper(h_value)],
            }
        )

    active_u_data = active_u_payload(active_u_raw_table(problem, support_modes, active_coeffs))
    denominator_data = interval_payload(denominator)
    offdiag_rows = []
    tasks = [(ordinal, i, j) for ordinal, (i, j) in enumerate(sorted(pairs), 1)]
    worker_count = resolve_workers(args.workers)
    print(f"off-diagonal entries: {len(tasks)}  workers={worker_count}  chunksize={args.chunksize}", flush=True)
    if worker_count == 1:
        init_offdiag_worker(problem, active_u_data, inactive_indices, records_by_pair, denominator_data, args.dps)
        iterator = map(offdiag_worker, tasks)
    else:
        context = multiprocessing.get_context("spawn")
        pool = context.Pool(
            processes=worker_count,
            initializer=init_offdiag_worker,
            initargs=(problem, active_u_data, inactive_indices, records_by_pair, denominator_data, args.dps),
        )
        iterator = pool.imap_unordered(offdiag_worker, tasks, chunksize=max(1, args.chunksize))
    processed = 0
    try:
        for ordinal, i, j, lower_value, upper_value, mid_value, radius_value in iterator:
            processed += 1
            if args.progress_every and processed % args.progress_every == 0:
                print(f"  offdiag {processed}/{len(tasks)}", flush=True)
            if radius_value < 0:
                raise RuntimeError(f"negative interval radius for pair {(i, j)}")
            mid[i, j] = mid[j, i] = mid_value
            rad[i, j] = rad[j, i] = radius_value
            if ordinal <= 10 or radius_value > 1e-5:
                offdiag_rows.append(
                    {
                        "i": i,
                        "j": j,
                        "interval": [lower_value, upper_value],
                        "mid": mid_value,
                        "radius": radius_value,
                    }
                )
    finally:
        if worker_count != 1:
            pool.close()
            pool.join()
    offdiag_rows.sort(key=lambda row: (row["i"], row["j"]))

    component_rows = []
    for component_id, component in enumerate(components):
        block_mid = mid[np.ix_(component, component)]
        block_rad = rad[np.ix_(component, component)]
        eigenvalues = np.linalg.eigvalsh(block_mid)
        row_sum_radius = float(np.max(np.sum(block_rad, axis=1)))
        frobenius_radius = float(np.linalg.norm(block_rad, ord="fro"))
        spectral_radius_bound = min(row_sum_radius, frobenius_radius)
        lower = float(eigenvalues[0] - spectral_radius_bound)
        component_rows.append(
            {
                "component_id": component_id,
                "dimension": len(component),
                "coordinates": [int(index) for index in component],
                "lambda_min_midpoint": float(eigenvalues[0]),
                "lambda_max_midpoint": float(eigenvalues[-1]),
                "row_sum_radius_bound": row_sum_radius,
                "frobenius_radius_bound": frobenius_radius,
                "spectral_radius_bound": spectral_radius_bound,
                "certified_lower_bound": lower,
                "passes": lower > 0.0,
            }
        )
    component_rows.sort(key=lambda row: row["certified_lower_bound"])
    passes = all(row["passes"] for row in component_rows)
    print(f"passes={passes} worst_lower={component_rows[0]['certified_lower_bound']:+.12e}")
    print(
        f"worst midpoint lambda={component_rows[0]['lambda_min_midpoint']:+.12e} "
        f"radius_bound={component_rows[0]['spectral_radius_bound']:.12e}"
    )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "interval_inactive_hessian_certificate" if passes else "interval_inactive_hessian_failed",
        "center": list(center),
        "radius": list(radius),
        "box": box.as_list(),
        "dps": args.dps,
        "workers": worker_count,
        "chunksize": args.chunksize,
        "active_map_diagnostics": map_diagnostics,
        "active_value_interval": [interval_lower(active_value), interval_upper(active_value)],
        "X2_interval": [interval_lower(x2), interval_upper(x2)],
        "D2_interval": [interval_lower(d2), interval_upper(d2)],
        "denominator_interval": [interval_lower(denominator), interval_upper(denominator)],
        "inactive_dimension": dimension,
        "inactive_modes": [int(index) for index in inactive_indices],
        "inactive_quadratic_triads": int(len(triad_indices)),
        "structural_pair_count": len(pairs),
        "component_count": len(components),
        "component_size_counts": dict(Counter(len(component) for component in components)),
        "passes": passes,
        "worst_component": component_rows[0],
        "components_by_certified_lower": component_rows,
        "max_entry_radius": float(np.max(rad)),
        "max_diagonal_radius": float(np.max(np.diag(rad))),
        "max_offdiagonal_radius": float(np.max(rad - np.diag(np.diag(rad)))),
        "diagonal_rows_sample": diagonal_rows[:20],
        "offdiag_rows_sample": offdiag_rows[:200],
        "elapsed_seconds": time.time() - start_time,
        "method": "Interval active map over reduced box; structural inactive Hessian entries; componentwise Weyl with min(row-sum radius, Frobenius radius).",
    }
    if not args.no_save:
        output = Path(args.output) if args.output else Path("results") / f"k3_reduced_bridge_interval_cert_{datetime.now():%Y%m%d_%H%M%S}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved: {output}")


if __name__ == "__main__":
    main()