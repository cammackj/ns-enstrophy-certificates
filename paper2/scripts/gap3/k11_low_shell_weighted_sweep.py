#!/usr/bin/env python3
"""Aggregate weighted Route A excess for low-representation k11 shells.

The angular-fiber obstruction is worst on shells with small r3.  This diagnostic
fixes a low-representation first shell n and sums the exact weighted pair probe
over every partner shell m in I_11.  The relevant question is whether the whole
n-row is close to the continuum coarea estimate, even if individual m entries
have large ratios.

This is diagnostic only: the Route A integral is evaluated numerically.
"""

from __future__ import annotations

import argparse
import math
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from k11_bottom_fiber_sweep import route_a_kernel
from k11_weighted_pair_probe import PairStats, weighted_pair_stats
from shell_rho_fiber_probe import lattice_shell, represented_shells


def default_worker_count(row_count: int) -> int:
    if row_count <= 1:
        return 1
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, row_count))


def route_a_integrand_vectorized(s_value: float, t_values: np.ndarray, rho_values: np.ndarray) -> np.ndarray:
    """Vectorized Route A joint integrand for one fixed first shell."""
    a_value = math.sqrt(s_value)
    b_values = np.sqrt(t_values)
    rho_shift = rho_values - s_value - t_values
    disc = 4.0 * s_value * t_values - rho_shift * rho_shift
    disc = np.maximum(disc, 0.0)
    s_minus_rho = s_value - rho_values
    s_minus_t = s_value - t_values
    rho_squared = rho_values * rho_values
    disc_squared = disc * disc
    a_cubed = s_value * a_value
    a_seventh = s_value * s_value * s_value * a_value
    b_cubed = t_values * b_values
    b_fifth = t_values * t_values * b_values
    b_seventh = t_values * t_values * t_values * b_values
    g_oa = s_minus_rho * s_minus_rho * disc_squared / (32.0 * a_seventh * b_seventh * rho_squared)
    g_pmid = s_minus_t * s_minus_t * disc / (8.0 * a_cubed * b_fifth * rho_squared)
    return g_oa + g_pmid


def route_a_kernel_vectorized(s_value: float, t_values: np.ndarray) -> np.ndarray:
    """Vectorized closed-form Route A joint kernel K(s,t)."""
    a_value = math.sqrt(s_value)
    b_values = np.sqrt(t_values)
    log2 = math.log(2.0)
    s = s_value
    t = t_values
    s2 = s * s
    s3 = s2 * s
    s4 = s2 * s2
    s5 = s4 * s
    s6 = s3 * s3
    t2 = t * t
    t3 = t2 * t
    t4 = t2 * t2
    s_minus_t = s - t
    s_minus_t2 = s_minus_t * s_minus_t
    a_cubed = s * a_value
    a_seventh = s3 * a_value
    b_fifth = t2 * b_values
    b_seventh = t3 * b_values
    p_oa = (
        5 * s6 - 20 * s5 * t + 30 * s4 * t2 + 150 * s4
        - 20 * s3 * t3 - 80 * s3 * t - 300 * s3
        + 5 * s2 * t4 + 40 * s2 * t2 - 120 * s2 * t + 350 * s2
        + 40 * s * t3 - 120 * s * t2 + 280 * s * t
        - 20 * s * s_minus_t2 * (3 * s2 + t2) * log2
        - 225 * s + 10 * t4 - 60 * t3 + 140 * t2 - 150 * t + 62
    )
    m_oa = p_oa / (320.0 * a_seventh * b_seventh)
    s_plus_t = s + t
    bracket = -1.0 + 2.0 * s_plus_t * log2 + 0.5 * (4.0 * s * t - s_plus_t * s_plus_t)
    m_pmid = s_minus_t2 / (8.0 * a_cubed * b_fifth) * bracket
    return m_oa + m_pmid


def prepare_block_modes(shells: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate all lattice modes in the supplied shell list."""
    mode_blocks = []
    shell_values = []
    shell_indices = []
    shell_counts = []
    for shell_index, shell in enumerate(shells):
        modes = lattice_shell(shell)
        mode_blocks.append(modes)
        shell_values.append(np.full(len(modes), shell, dtype=np.int64))
        shell_indices.append(np.full(len(modes), shell_index, dtype=np.int32))
        shell_counts.append(len(modes))
    return (
        np.vstack(mode_blocks),
        np.concatenate(shell_values),
        np.concatenate(shell_indices),
        np.asarray(shell_counts, dtype=np.float64),
    )


def cubic_orbit_representatives(shell: int) -> list[tuple[np.ndarray, int]]:
    """Return one representative per signed-permutation orbit on a shell."""
    orbit_sizes: Counter[tuple[int, int, int]] = Counter()
    for mode in lattice_shell(shell):
        key = tuple(sorted(abs(int(coord)) for coord in mode))
        orbit_sizes[key] += 1
    return [(np.array(key, dtype=np.int64), size) for key, size in sorted(orbit_sizes.items())]


def sweep_shell(
    k: int,
    n_shell: int,
    partner_shells: list[int],
    progress_every: int = 0,
) -> tuple[float, float, float, float, PairStats]:
    actual = 0.0
    continuum = 0.0
    positive_excess = 0.0
    worst: PairStats | None = None

    for partner_index, m_shell in enumerate(partner_shells, start=1):
        row = weighted_pair_stats(k, n_shell, m_shell)
        actual += row.actual
        continuum += row.continuum
        positive_excess += max(0.0, row.actual - row.continuum)
        if worst is None or row.ratio > worst.ratio:
            worst = row
        if progress_every > 0 and partner_index % progress_every == 0:
            current_ratio = actual / continuum if continuum > 0 else 0.0
            print(
                f"      n={n_shell} partners={partner_index}/{len(partner_shells)} "
                f"partial_ratio={current_ratio:.6f}",
                flush=True,
            )

    if worst is None:
        raise RuntimeError("no partner shells were inspected")
    ratio = actual / continuum if continuum > 0 else 0.0
    return ratio, actual, continuum, positive_excess, worst


def sweep_shell_fast(
    k: int,
    n_shell: int,
    partner_shells: list[int],
    block_modes: np.ndarray,
    block_shell_values: np.ndarray,
    block_shell_indices: np.ndarray,
    block_shell_counts: np.ndarray,
    progress_every: int = 0,
) -> tuple[float, float, float, float, PairStats]:
    """Sweep one full row by scanning all block modes at once for each p."""
    scale = 2**k
    lo = scale
    hi = 2 * scale
    r3_n = len(lattice_shell(n_shell))
    p_orbits = cubic_orbit_representatives(n_shell)
    s_value = n_shell / scale
    actual_by_shell = np.zeros(len(partner_shells), dtype=np.float64)

    for p_index, (p_mode, orbit_size) in enumerate(p_orbits, start=1):
        dots = block_modes @ p_mode
        ell_values = n_shell + block_shell_values + 2 * dots
        mask = (ell_values >= lo) & (ell_values < hi)
        mask &= ~((block_shell_values == n_shell) & (ell_values == n_shell))
        if np.any(mask):
            t_values = block_shell_values[mask].astype(np.float64) / scale
            rho_values = ell_values[mask].astype(np.float64) / scale
            weights = route_a_integrand_vectorized(s_value, t_values, rho_values)
            actual_by_shell += np.bincount(
                block_shell_indices[mask],
                weights=weights * orbit_size,
                minlength=len(partner_shells),
            )
        if progress_every > 0 and p_index % progress_every == 0:
            print(
                f"      n={n_shell} orbits={p_index}/{len(p_orbits)}",
                flush=True,
            )

    partner_shell_array = np.asarray(partner_shells, dtype=np.float64)
    continuum_by_shell = (
        r3_n
        * block_shell_counts
        * scale
        * route_a_kernel_vectorized(s_value, partner_shell_array / scale)
        / (4.0 * np.sqrt(n_shell * partner_shell_array))
    )
    ratios = np.divide(
        actual_by_shell,
        continuum_by_shell,
        out=np.zeros_like(actual_by_shell),
        where=continuum_by_shell > 0,
    )
    worst_index = int(np.argmax(ratios))
    actual = float(actual_by_shell.sum())
    continuum = float(continuum_by_shell.sum())
    positive_excess = float(np.maximum(actual_by_shell - continuum_by_shell, 0.0).sum())
    ratio = actual / continuum if continuum > 0 else 0.0
    worst = PairStats(
        n=n_shell,
        m=partner_shells[worst_index],
        r3_n=r3_n,
        r3_m=int(block_shell_counts[worst_index]),
        active_pairs=0,
        actual=float(actual_by_shell[worst_index]),
        continuum=float(continuum_by_shell[worst_index]),
        ratio=float(ratios[worst_index]),
        max_fiber=0,
    )
    return ratio, actual, continuum, positive_excess, worst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=11)
    parser.add_argument("--max-r3", type=int, default=24)
    parser.add_argument("--min-r3", type=int, default=0)
    parser.add_argument("--all", action="store_true", help="inspect every represented first-shell row")
    parser.add_argument(
        "--shell",
        action="append",
        type=int,
        default=[],
        help="explicit first-shell row to inspect; may be repeated",
    )
    parser.add_argument("--limit", type=int, help="limit the number of low shells inspected")
    parser.add_argument("--progress-every", type=int, default=0, help="print progress after this many partner shells")
    parser.add_argument("--fast", action="store_true", help="use whole-row block aggregation instead of pair-by-pair aggregation")
    parser.add_argument("--quiet-rows", action="store_true", help="suppress per-row output and print only the sorted summary")
    parser.add_argument("--top", type=int, help="limit the sorted summary to the top N rows")
    parser.add_argument("--progress-rows", type=int, default=0, help="print progress after this many first-shell rows")
    parser.add_argument("--workers", type=int, default=0, help="row-level worker threads; 0 uses all logical CPUs")
    args = parser.parse_args()

    shells = represented_shells(args.k)
    counts = {shell: len(lattice_shell(shell)) for shell in shells}
    if args.all:
        low_shells = shells
    elif args.shell:
        missing = [shell for shell in args.shell if shell not in counts]
        if missing:
            raise ValueError(f"requested shells are not represented in I_{args.k}: {missing}")
        low_shells = args.shell
    else:
        low_shells = [shell for shell in shells if args.min_r3 <= counts[shell] <= args.max_r3]
    if args.limit is not None:
        low_shells = low_shells[: args.limit]

    label = "all shells" if args.all else "explicit shells" if args.shell else f"r3 in [{args.min_r3},{args.max_r3}]"
    print(f"k={args.k}, {label}, inspected shells={len(low_shells)}")
    if not args.quiet_rows:
        print(f"shell list: {[(shell, counts[shell]) for shell in low_shells]}")
    if not args.quiet_rows:
        print("    n  r3_n  aggregate       actual    continuum  pos_excess worst_m worst_ratio")

    block_data = prepare_block_modes(shells) if args.fast else None
    worker_count = args.workers if args.workers > 0 else default_worker_count(len(low_shells))
    print(f"row workers={worker_count}")

    def inspect_row(n_shell: int) -> tuple[float, int, int, float, float, float, PairStats]:
        if block_data is None:
            ratio, actual, continuum, positive_excess, worst = sweep_shell(
                args.k,
                n_shell,
                shells,
                progress_every=args.progress_every if worker_count == 1 else 0,
            )
        else:
            ratio, actual, continuum, positive_excess, worst = sweep_shell_fast(
                args.k,
                n_shell,
                shells,
                *block_data,
                progress_every=args.progress_every if worker_count == 1 else 0,
            )
        return ratio, n_shell, counts[n_shell], actual, continuum, positive_excess, worst

    rows = []
    if worker_count == 1:
        row_results = (inspect_row(n_shell) for n_shell in low_shells)
        for row_index, row_result in enumerate(row_results, start=1):
            ratio, n_shell, r3_count, actual, continuum, positive_excess, worst = row_result
            rows.append(row_result)
            if not args.quiet_rows:
                print(
                    f"{n_shell:5d} {r3_count:5d} {ratio:10.6f} "
                    f"{actual:12.3f} {continuum:12.3f} {positive_excess:11.3f} "
                    f"{worst.m:7d} {worst.ratio:11.6f}",
                    flush=True,
                )
            elif args.progress_rows > 0 and row_index % args.progress_rows == 0:
                best_ratio, best_shell, best_r3, *_ = max(rows)
                print(
                    f"      rows={row_index}/{len(low_shells)} current_best={best_ratio:.6f} "
                    f"at n={best_shell} r3={best_r3}",
                    flush=True,
                )
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(inspect_row, n_shell) for n_shell in low_shells]
            for row_index, future in enumerate(as_completed(futures), start=1):
                row_result = future.result()
                ratio, n_shell, r3_count, actual, continuum, positive_excess, worst = row_result
                rows.append(row_result)
                if not args.quiet_rows:
                    print(
                        f"{n_shell:5d} {r3_count:5d} {ratio:10.6f} "
                        f"{actual:12.3f} {continuum:12.3f} {positive_excess:11.3f} "
                        f"{worst.m:7d} {worst.ratio:11.6f}",
                        flush=True,
                    )
                elif args.progress_rows > 0 and row_index % args.progress_rows == 0:
                    best_ratio, best_shell, best_r3, *_ = max(rows)
                    print(
                        f"      rows={row_index}/{len(low_shells)} current_best={best_ratio:.6f} "
                        f"at n={best_shell} r3={best_r3}",
                        flush=True,
                    )

    print("\nSorted by aggregate ratio")
    sorted_rows = sorted(rows, reverse=True)
    if args.top is not None:
        sorted_rows = sorted_rows[: args.top]
    for ratio, n_shell, r3_count, actual, continuum, positive_excess, worst in sorted_rows:
        print(
            f"n={n_shell} r3={r3_count} aggregate={ratio:.6f} "
            f"pos_excess={positive_excess:.3f} worst_m={worst.m} worst_ratio={worst.ratio:.6f}"
        )


if __name__ == "__main__":
    main()