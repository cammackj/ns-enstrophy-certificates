#!/usr/bin/env python3
"""Fast guarded row certificate for the midpoint-centered kernel.

This complements `centered_row_mp_verify.py`.  It uses the same orbit-reduced
exact integer row scan as the probes, but applies explicit absolute allowances
to the centered point integrand and centered continuum kernel:

    A_upper = sum_pairs (g_center_float + eps_g)
    C_lower = sum_shells factor(n,m) * max(K_center_float - eps_k, 0)

The default allowances are intentionally much larger than ordinary float64
roundoff.  Rows near the requested budget should be rechecked with
`centered_row_mp_verify.py --interval`.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from k11_low_shell_weighted_sweep import cubic_orbit_representatives, prepare_block_modes
from shell_rho_fiber_probe import lattice_shell, represented_shells


def default_worker_count(row_count: int) -> int:
    return max(1, min(os.cpu_count() or 1, row_count))


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


def poly_mul_vectorized(left: list[np.ndarray], right: list[np.ndarray]) -> list[np.ndarray]:
    zero = np.zeros_like(left[0], dtype=np.float64)
    out = [zero.copy() for _ in range(len(left) + len(right) - 1)]
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            out[left_index + right_index] += left_value * right_value
    return out


def centered_integrand_vectorized(s_value: float, t_values: np.ndarray, rho_values: np.ndarray) -> np.ndarray:
    st_values = s_value * t_values
    rho_shift = rho_values - s_value - t_values
    disc = np.maximum(4.0 * st_values - rho_shift * rho_shift, 0.0)
    centered = rho_values - 1.5
    denom = 32.0 * st_values**3 * np.sqrt(st_values) * rho_values * rho_values
    return centered * centered * disc * disc / denom


def centered_kernel_vectorized(s_value: float, t_values: np.ndarray) -> np.ndarray:
    """Closed-form integral of the centered kernel over 1 <= rho <= 2."""
    t_values = np.asarray(t_values, dtype=np.float64)
    st_values = s_value * t_values
    u_values = s_value + t_values
    constant = 4.0 * st_values - u_values * u_values
    disc = [constant, 2.0 * u_values, -np.ones_like(t_values, dtype=np.float64)]
    centered = [2.25 * np.ones_like(t_values, dtype=np.float64), -3.0 * np.ones_like(t_values, dtype=np.float64), np.ones_like(t_values, dtype=np.float64)]
    numerator = poly_mul_vectorized(centered, poly_mul_vectorized(disc, disc))
    integral = np.zeros_like(t_values, dtype=np.float64)
    for power, coeff in enumerate(numerator):
        exponent = power - 2
        if exponent == -2:
            integral += 0.5 * coeff
        elif exponent == -1:
            integral += math.log(2.0) * coeff
        else:
            integral += coeff * ((2.0 ** (exponent + 1)) - 1.0) / (exponent + 1)
    return integral / (32.0 * st_values**3 * np.sqrt(st_values))


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

    for orbit_index, (p_mode, orbit_size) in enumerate(cubic_orbit_representatives(n_shell), start=1):
        dots = block_modes @ p_mode
        ell_values = n_shell + block_shell_values + 2 * dots
        mask = (ell_values >= scale) & (ell_values < 2 * scale)
        mask &= ~((block_shell_values == n_shell) & (ell_values == n_shell))
        if np.any(mask):
            t_values = block_shell_values[mask].astype(np.float64) / scale
            rho_values = ell_values[mask].astype(np.float64) / scale
            weights = centered_integrand_vectorized(s_value, t_values, rho_values)
            pair_count = int(np.count_nonzero(mask)) * orbit_size
            active_pairs += pair_count
            actual_upper_chunks.append(float(orbit_size) * sum_upper(weights + eps_g))
        if progress_every > 0 and orbit_index % progress_every == 0:
            print(f"      n={n_shell} orbits={orbit_index}", flush=True)

    actual_upper = sum_upper(np.asarray(actual_upper_chunks, dtype=np.float64))
    shell_array = np.asarray(shells, dtype=np.float64)
    r3_array = np.asarray([shell_counts[shell] for shell in shells], dtype=np.float64)
    factors = shell_counts[n_shell] * r3_array * scale / (4.0 * np.sqrt(n_shell * shell_array))
    kernel_values = centered_kernel_vectorized(s_value, shell_array / scale)
    continuum_lower = sum_lower(factors * np.maximum(kernel_values - eps_k, 0.0))
    ratio_upper = actual_upper / continuum_lower if continuum_lower > 0 else math.inf
    return ratio_upper, actual_upper, continuum_lower, active_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--shell", action="append", type=int, default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--min-r3", type=int, default=0)
    parser.add_argument("--max-r3", type=int)
    parser.add_argument("--eps-g", type=float, default=1e-8)
    parser.add_argument("--eps-k", type=float, default=1e-10)
    parser.add_argument("--budget-ratio", type=float, default=1.5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--progress-rows", type=int, default=0)
    parser.add_argument("--top", type=int, default=20)
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
        rows = [shells[0]]
    missing = [shell for shell in rows if shell not in shell_counts]
    if missing:
        raise ValueError(f"requested shells are not represented in I_{args.k}: {missing}")

    print(f"k={args.k}, rows={len(rows)}, eps_g={args.eps_g:g}, eps_k={args.eps_k:g}, budget={args.budget_ratio}")
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

    print("\nSorted by coarse centered upper ratio")
    for ratio_upper, n_shell, r3_count, actual_upper, continuum_lower, active_pairs in sorted(results, reverse=True)[: args.top]:
        print(
            f"n={n_shell} r3={r3_count} pairs={active_pairs} ratio_upper={ratio_upper:.12f} "
            f"actual_upper={actual_upper:.12g} continuum_lower={continuum_lower:.12g} "
            f"budget_margin={args.budget_ratio - ratio_upper:.12f}"
        )


if __name__ == "__main__":
    main()