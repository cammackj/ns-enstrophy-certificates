#!/usr/bin/env python3
"""Vectorized interval screen for k11 weighted row sums.

This is a faster finite-certificate front end than scalar mpmath intervals.  It
uses one signed-permutation orbit representative at a time and evaluates the
Route A weights with simple outward-rounded float64 interval arithmetic.  The
optional `--counted-cells` mode first groups exact `(m, ell)` row counts from
`k11_row_mp_verify`; the default direct mode is faster for all-row screening.
Rows close to the budget should still be confirmed by the mpmath interval
verifier, but this screen is designed to make the all-row certificate practical.
"""

from __future__ import annotations

import argparse
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from k11_row_mp_verify import cubic_orbit_representatives, prepare_block, row_counts
from shell_rho_fiber_probe import lattice_shell, represented_shells


Interval = tuple[np.ndarray, np.ndarray]


def default_worker_count(row_count: int) -> int:
    if row_count <= 1:
        return 1
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, row_count))


def down(values: np.ndarray | float) -> np.ndarray:
    return np.nextafter(np.asarray(values, dtype=np.float64), -np.inf)


def up(values: np.ndarray | float) -> np.ndarray:
    return np.nextafter(np.asarray(values, dtype=np.float64), np.inf)


def point(values: np.ndarray | float) -> Interval:
    array = np.asarray(values, dtype=np.float64)
    return down(array), up(array)


def const(value: float) -> Interval:
    return point(value)


def add(left: Interval, right: Interval) -> Interval:
    return down(left[0] + right[0]), up(left[1] + right[1])


def sub(left: Interval, right: Interval) -> Interval:
    return down(left[0] - right[1]), up(left[1] - right[0])


def mul(left: Interval, right: Interval) -> Interval:
    products = np.stack(
        [left[0] * right[0], left[0] * right[1], left[1] * right[0], left[1] * right[1]],
        axis=0,
    )
    return down(np.min(products, axis=0)), up(np.max(products, axis=0))


def reciprocal(value: Interval) -> Interval:
    if np.any(value[0] <= 0):
        raise ValueError("interval reciprocal requires positive lower endpoint")
    return down(1.0 / value[1]), up(1.0 / value[0])


def div(left: Interval, right: Interval) -> Interval:
    return mul(left, reciprocal(right))


def sqrt_interval(value: Interval) -> Interval:
    if np.any(value[1] < 0):
        raise ValueError("sqrt interval has negative upper endpoint")
    return down(np.sqrt(np.maximum(value[0], 0.0))), up(np.sqrt(np.maximum(value[1], 0.0)))


def pow_int(value: Interval, exponent: int) -> Interval:
    if exponent == 0:
        return const(1.0)
    result = value
    for _ in range(exponent - 1):
        result = mul(result, value)
    return result


def scale_interval(value: Interval, factor: float) -> Interval:
    return mul(value, const(factor))


def positive_sum_upper(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    total = float(np.sum(values, dtype=np.float64))
    gamma = values.size * np.finfo(np.float64).eps
    if gamma >= 0.5:
        gamma = 0.5
    return float(up(total * (1.0 + gamma / (1.0 - gamma))))


def positive_sum_lower(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    total = float(np.sum(values, dtype=np.float64))
    gamma = values.size * np.finfo(np.float64).eps
    if gamma >= 0.5:
        gamma = 0.5
    return float(down(total * (1.0 - gamma / (1.0 - gamma))))


def route_a_integrand_interval_upper(k: int, n_shell: int, m_shells: np.ndarray, ell_shells: np.ndarray) -> np.ndarray:
    scale = float(2**k)
    s = point(n_shell / scale)
    t = point(m_shells.astype(np.float64) / scale)
    rho = point(ell_shells.astype(np.float64) / scale)
    a = sqrt_interval(s)
    b = sqrt_interval(t)
    disc = sub(scale_interval(mul(s, t), 4.0), pow_int(sub(sub(rho, s), t), 2))
    g_oa_num = mul(pow_int(sub(s, rho), 2), pow_int(disc, 2))
    g_oa_den = scale_interval(mul(mul(pow_int(a, 7), pow_int(b, 7)), pow_int(rho, 2)), 32.0)
    g_pmid_num = mul(pow_int(sub(s, t), 2), disc)
    g_pmid_den = scale_interval(mul(mul(pow_int(a, 3), pow_int(b, 5)), pow_int(rho, 2)), 8.0)
    return add(div(g_oa_num, g_oa_den), div(g_pmid_num, g_pmid_den))[1]


def route_a_kernel_interval(k: int, n_shell: int, m_shells: np.ndarray) -> Interval:
    scale = float(2**k)
    s = point(n_shell / scale)
    t = point(m_shells.astype(np.float64) / scale)
    a = sqrt_interval(s)
    b = sqrt_interval(t)
    log2 = point(math.log(2.0))

    terms = add(scale_interval(pow_int(s, 6), 5.0), scale_interval(mul(pow_int(s, 5), t), -20.0))
    for term in [
        scale_interval(mul(pow_int(s, 4), pow_int(t, 2)), 30.0),
        scale_interval(pow_int(s, 4), 150.0),
        scale_interval(mul(pow_int(s, 3), pow_int(t, 3)), -20.0),
        scale_interval(mul(pow_int(s, 3), t), -80.0),
        scale_interval(pow_int(s, 3), -300.0),
        scale_interval(mul(pow_int(s, 2), pow_int(t, 4)), 5.0),
        scale_interval(mul(pow_int(s, 2), pow_int(t, 2)), 40.0),
        scale_interval(mul(pow_int(s, 2), t), -120.0),
        scale_interval(pow_int(s, 2), 350.0),
        scale_interval(mul(s, pow_int(t, 3)), 40.0),
        scale_interval(mul(s, pow_int(t, 2)), -120.0),
        scale_interval(mul(s, t), 280.0),
        scale_interval(mul(mul(mul(s, pow_int(sub(s, t), 2)), add(scale_interval(pow_int(s, 2), 3.0), pow_int(t, 2))), log2), -20.0),
        scale_interval(s, -225.0),
        scale_interval(pow_int(t, 4), 10.0),
        scale_interval(pow_int(t, 3), -60.0),
        scale_interval(pow_int(t, 2), 140.0),
        scale_interval(t, -150.0),
        const(62.0),
    ]:
        terms = add(terms, term)
    m_oa = div(terms, scale_interval(mul(pow_int(a, 7), pow_int(b, 7)), 320.0))

    bracket = add(
        const(-1.0),
        add(
            scale_interval(mul(add(s, t), log2), 2.0),
            scale_interval(sub(scale_interval(mul(s, t), 4.0), pow_int(add(s, t), 2)), 0.5),
        ),
    )
    m_pmid = mul(
        div(pow_int(sub(s, t), 2), scale_interval(mul(pow_int(a, 3), pow_int(b, 5)), 8.0)),
        bracket,
    )
    return add(m_oa, m_pmid)


def continuum_lower_for_row(k: int, n_shell: int, shells: list[int], shell_counts: dict[int, int]) -> float:
    shell_array = np.asarray(shells, dtype=np.float64)
    kernel_lower, _ = route_a_kernel_interval(k, n_shell, shell_array.astype(np.int64))
    if np.any(kernel_lower <= 0):
        raise ValueError(f"row {n_shell} has nonpositive kernel lower endpoint")
    continuum_terms = down(
        shell_counts[n_shell]
        * np.asarray([shell_counts[shell] for shell in shells], dtype=np.float64)
        * (2**k)
        * kernel_lower
        / up(4.0 * np.sqrt(n_shell * shell_array))
    )
    return positive_sum_lower(continuum_terms)


def interval_row_ratio_counted(
    k: int,
    n_shell: int,
    shells: list[int],
    shell_counts: dict[int, int],
    block_modes: np.ndarray,
    block_shell_values: np.ndarray,
    block_shell_indices: np.ndarray,
    progress_every: int,
) -> tuple[float, float, float, int]:
    counts = row_counts(k, n_shell, shells, block_modes, block_shell_values, block_shell_indices, progress_every)
    cells = np.array([(m_shell, ell_shell, count) for (m_shell, ell_shell), count in counts.items()], dtype=np.int64)
    if len(cells) == 0:
        return 0.0, 0.0, 0.0, 0

    weights_upper = route_a_integrand_interval_upper(k, n_shell, cells[:, 0], cells[:, 1])
    actual_terms = up(cells[:, 2].astype(np.float64) * weights_upper)
    actual_upper = positive_sum_upper(actual_terms)
    continuum_lower = continuum_lower_for_row(k, n_shell, shells, shell_counts)
    return actual_upper / continuum_lower, actual_upper, continuum_lower, len(cells)


def interval_row_ratio_direct(
    k: int,
    n_shell: int,
    shells: list[int],
    shell_counts: dict[int, int],
    block_modes: np.ndarray,
    block_shell_values: np.ndarray,
    _block_shell_indices: np.ndarray,
    progress_every: int,
) -> tuple[float, float, float, int]:
    scale = 2**k
    orbit_upper_sums = []
    active_pairs = 0
    p_orbits = cubic_orbit_representatives(n_shell)
    for orbit_index, (p_mode, orbit_size) in enumerate(p_orbits, start=1):
        dots = block_modes @ p_mode
        ell_values = n_shell + block_shell_values + 2 * dots
        mask = (ell_values >= scale) & (ell_values < 2 * scale)
        mask &= ~((block_shell_values == n_shell) & (ell_values == n_shell))
        if np.any(mask):
            weights_upper = route_a_integrand_interval_upper(k, n_shell, block_shell_values[mask], ell_values[mask])
            orbit_terms = up(float(orbit_size) * weights_upper)
            orbit_upper_sums.append(positive_sum_upper(orbit_terms))
            active_pairs += int(np.count_nonzero(mask)) * orbit_size
        if progress_every > 0 and orbit_index % progress_every == 0:
            print(f"      counted n={n_shell} orbits={orbit_index}/{len(p_orbits)}", flush=True)

    if not orbit_upper_sums:
        return 0.0, 0.0, 0.0, 0
    actual_upper = positive_sum_upper(np.asarray(orbit_upper_sums, dtype=np.float64))
    continuum_lower = continuum_lower_for_row(k, n_shell, shells, shell_counts)
    return actual_upper / continuum_lower, actual_upper, continuum_lower, active_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=11)
    parser.add_argument("--shell", action="append", type=int, default=[])
    parser.add_argument("--min-r3", type=int, default=0)
    parser.add_argument("--max-r3", type=int)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--budget-ratio", type=float, default=1.1503561001915947)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--progress-rows", type=int, default=0)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--workers", type=int, default=0, help="row-level worker threads; 0 uses all logical CPUs")
    parser.add_argument("--counted-cells", action="store_true", help="group exact (m,ell) cells before interval evaluation")
    args = parser.parse_args()

    shells = represented_shells(args.k)
    shell_counts = {shell: len(lattice_shell(shell)) for shell in shells}
    if args.all:
        rows = shells
    elif args.shell:
        rows = args.shell
    elif args.max_r3 is not None:
        rows = [shell for shell in shells if args.min_r3 <= shell_counts[shell] <= args.max_r3]
    else:
        rows = [2048]
    missing = [shell for shell in rows if shell not in shell_counts]
    if missing:
        raise ValueError(f"requested shells are not represented in I_{args.k}: {missing}")

    print(f"k={args.k}, interval-screen rows={len(rows)}, budget={args.budget_ratio}")
    block_modes, block_shell_values, block_shell_indices = prepare_block(shells)
    row_ratio = interval_row_ratio_counted if args.counted_cells else interval_row_ratio_direct
    worker_count = args.workers if args.workers > 0 else default_worker_count(len(rows))
    print(f"row workers={worker_count}")

    def run_row(n_shell: int) -> tuple[float, int, int, float, float, int]:
        ratio, actual_upper, continuum_lower, item_count = row_ratio(
            args.k,
            n_shell,
            shells,
            shell_counts,
            block_modes,
            block_shell_values,
            block_shell_indices,
            args.progress_every if worker_count == 1 else 0,
        )
        return ratio, n_shell, shell_counts[n_shell], actual_upper, continuum_lower, item_count

    results = []
    if worker_count == 1:
        row_results = (run_row(n_shell) for n_shell in rows)
        for row_index, row_result in enumerate(row_results, start=1):
            ratio, n_shell, r3_count, actual_upper, continuum_lower, item_count = row_result
            if ratio >= args.budget_ratio:
                raise SystemExit(f"row {n_shell} exceeds budget: ratio_upper={ratio}")
            results.append(row_result)
            if args.progress_rows > 0 and row_index % args.progress_rows == 0:
                best = max(results)
                print(f"      rows={row_index}/{len(rows)} best={best[0]:.12f} at n={best[1]} r3={best[2]}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(run_row, n_shell) for n_shell in rows]
            for row_index, future in enumerate(as_completed(futures), start=1):
                row_result = future.result()
                ratio, n_shell, r3_count, actual_upper, continuum_lower, item_count = row_result
                if ratio >= args.budget_ratio:
                    raise SystemExit(f"row {n_shell} exceeds budget: ratio_upper={ratio}")
                results.append(row_result)
                if args.progress_rows > 0 and row_index % args.progress_rows == 0:
                    best = max(results)
                    print(f"      rows={row_index}/{len(rows)} best={best[0]:.12f} at n={best[1]} r3={best[2]}", flush=True)

    print("\nSorted by interval upper ratio")
    item_label = "cells" if args.counted_cells else "pairs"
    for ratio, n_shell, r3_count, actual_upper, continuum_lower, item_count in sorted(results, reverse=True)[: args.top]:
        print(
            f"n={n_shell} r3={r3_count} {item_label}={item_count} ratio_upper={ratio:.15f} "
            f"actual_upper={actual_upper:.12g} continuum_lower={continuum_lower:.12g} "
            f"budget_margin={args.budget_ratio - ratio:.12f}"
        )


if __name__ == "__main__":
    main()