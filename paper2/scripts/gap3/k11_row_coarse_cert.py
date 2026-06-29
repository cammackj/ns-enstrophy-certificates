#!/usr/bin/env python3
"""Fast coarse certificate screen for k11 weighted row sums.

This script complements the exact interval row verifier.  It performs the fast
orbit-reduced row scan, then applies explicit absolute error allowances to the
joint integrand `g` and continuum kernel `K`:

    A_upper = sum_pairs (g_float + eps_g)
    C_lower = sum_shells factor(n,m) * max(K_float - eps_K, 0)

The default allowances are deliberately much larger than ordinary float64
roundoff.  Rows close to the budget should still be checked by
`k11_row_mp_verify.py --interval`.
"""

from __future__ import annotations

import argparse
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from k11_low_shell_weighted_sweep import (
    cubic_orbit_representatives,
    prepare_block_modes,
    route_a_integrand_vectorized,
    route_a_kernel_vectorized,
)
from shell_rho_fiber_probe import lattice_shell, represented_shells


def default_worker_count(row_count: int) -> int:
    if row_count <= 1:
        return 1
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, row_count))


def sum_upper(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    total = float(np.sum(values, dtype=np.float64))
    gamma = values.size * np.finfo(np.float64).eps
    if gamma >= 0.5:
        gamma = 0.5
    return float(np.nextafter(total * (1.0 + gamma / (1.0 - gamma)), math.inf))


def sum_lower(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    total = float(np.sum(values, dtype=np.float64))
    gamma = values.size * np.finfo(np.float64).eps
    if gamma >= 0.5:
        gamma = 0.5
    return float(np.nextafter(total * (1.0 - gamma / (1.0 - gamma)), -math.inf))


def coarse_row_bound(
    k: int,
    n_shell: int,
    shells: list[int],
    shell_counts: dict[int, int],
    block_modes: np.ndarray,
    block_shell_values: np.ndarray,
    eps_g: float,
    eps_k: float,
    progress_every: int,
) -> tuple[float, float, float, int]:
    scale = 2**k
    s_value = n_shell / scale
    actual_upper_chunks = []
    active_pairs = 0
    p_orbits = cubic_orbit_representatives(n_shell)

    for orbit_index, (p_mode, orbit_size) in enumerate(p_orbits, start=1):
        dots = block_modes @ p_mode
        ell_values = n_shell + block_shell_values + 2 * dots
        mask = (ell_values >= scale) & (ell_values < 2 * scale)
        mask &= ~((block_shell_values == n_shell) & (ell_values == n_shell))
        if np.any(mask):
            t_values = block_shell_values[mask].astype(np.float64) / scale
            rho_values = ell_values[mask].astype(np.float64) / scale
            weights = route_a_integrand_vectorized(s_value, t_values, rho_values)
            pair_count = int(np.count_nonzero(mask)) * orbit_size
            active_pairs += pair_count
            actual_upper_chunks.append(float(orbit_size) * sum_upper(weights + eps_g))
        if progress_every > 0 and orbit_index % progress_every == 0:
            print(f"      n={n_shell} orbits={orbit_index}/{len(p_orbits)}", flush=True)

    actual_upper = sum_upper(np.asarray(actual_upper_chunks, dtype=np.float64))
    shell_array = np.asarray(shells, dtype=np.float64)
    r3_array = np.asarray([shell_counts[shell] for shell in shells], dtype=np.float64)
    factors = shell_counts[n_shell] * r3_array * scale / (4.0 * np.sqrt(n_shell * shell_array))
    kernel_values = route_a_kernel_vectorized(s_value, shell_array / scale)
    continuum_lower = sum_lower(factors * np.maximum(kernel_values - eps_k, 0.0))
    ratio_upper = actual_upper / continuum_lower if continuum_lower > 0 else math.inf
    return ratio_upper, actual_upper, continuum_lower, active_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=11)
    parser.add_argument("--shell", action="append", type=int, default=[])
    parser.add_argument("--min-r3", type=int, default=0)
    parser.add_argument("--max-r3", type=int)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--eps-g", type=float, default=1e-4)
    parser.add_argument("--eps-k", type=float, default=1e-6)
    parser.add_argument("--budget-ratio", type=float, default=1.1503561001915947)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--progress-rows", type=int, default=0)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--workers", type=int, default=0, help="row-level worker threads; 0 uses all logical CPUs")
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

    print(
        f"k={args.k}, rows={len(rows)}, eps_g={args.eps_g:g}, eps_k={args.eps_k:g}, "
        f"budget={args.budget_ratio}"
    )
    block_modes, block_shell_values, _, _ = prepare_block_modes(shells)
    worker_count = args.workers if args.workers > 0 else default_worker_count(len(rows))
    print(f"row workers={worker_count}")

    def run_row(n_shell: int) -> tuple[float, int, int, float, float, int]:
        ratio_upper, actual_upper, continuum_lower, active_pairs = coarse_row_bound(
            args.k,
            n_shell,
            shells,
            shell_counts,
            block_modes,
            block_shell_values,
            args.eps_g,
            args.eps_k,
            args.progress_every if worker_count == 1 else 0,
        )
        return ratio_upper, n_shell, shell_counts[n_shell], actual_upper, continuum_lower, active_pairs

    results = []
    if worker_count == 1:
        row_results = (run_row(n_shell) for n_shell in rows)
        for row_index, row_result in enumerate(row_results, start=1):
            ratio_upper, n_shell, r3_count, actual_upper, continuum_lower, active_pairs = row_result
            if ratio_upper >= args.budget_ratio:
                raise SystemExit(f"row {n_shell} exceeds budget: coarse ratio_upper={ratio_upper}")
            results.append(row_result)
            if args.progress_rows > 0 and row_index % args.progress_rows == 0:
                best = max(results)
                print(f"      rows={row_index}/{len(rows)} best={best[0]:.9f} at n={best[1]} r3={best[2]}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(run_row, n_shell) for n_shell in rows]
            for row_index, future in enumerate(as_completed(futures), start=1):
                row_result = future.result()
                ratio_upper, n_shell, r3_count, actual_upper, continuum_lower, active_pairs = row_result
                if ratio_upper >= args.budget_ratio:
                    raise SystemExit(f"row {n_shell} exceeds budget: coarse ratio_upper={ratio_upper}")
                results.append(row_result)
                if args.progress_rows > 0 and row_index % args.progress_rows == 0:
                    best = max(results)
                    print(f"      rows={row_index}/{len(rows)} best={best[0]:.9f} at n={best[1]} r3={best[2]}", flush=True)

    print("\nSorted by coarse upper ratio")
    for ratio_upper, n_shell, r3_count, actual_upper, continuum_lower, active_pairs in sorted(results, reverse=True)[: args.top]:
        print(
            f"n={n_shell} r3={r3_count} pairs={active_pairs} ratio_upper={ratio_upper:.12f} "
            f"actual_upper={actual_upper:.12g} continuum_lower={continuum_lower:.12g} "
            f"budget_margin={args.budget_ratio - ratio_upper:.12f}"
        )


if __name__ == "__main__":
    main()