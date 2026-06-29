#!/usr/bin/env python3
"""High-precision verifier for k11 weighted row sums.

This script is the next step after the floating weighted-row diagnostics.  It
first aggregates exact integer counts for each `(m, ell)` shell/rho cell in a
fixed first-shell row `n`, then evaluates the Route A weights with mpmath.  In
`--interval` mode it reports an upper bound for the actual row divided by a
lower bound for the continuum row.
"""

from __future__ import annotations

import argparse
import math
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import mpmath as mp
import numpy as np

from shell_rho_fiber_probe import lattice_shell, represented_shells


def default_worker_count(row_count: int) -> int:
    if row_count <= 1:
        return 1
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, row_count))


def route_a_integrand_mp(k: int, n_shell: int, m_shell: int, ell_shell: int) -> mp.mpf:
    """Route A joint integrand g(s,t,rho) at high precision."""
    scale = mp.mpf(2) ** k
    s_value = mp.mpf(n_shell) / scale
    t_value = mp.mpf(m_shell) / scale
    rho_value = mp.mpf(ell_shell) / scale
    a_value = mp.sqrt(s_value)
    b_value = mp.sqrt(t_value)
    disc = 4 * s_value * t_value - (rho_value - s_value - t_value) ** 2
    if disc < 0 and abs(disc) < mp.mpf("1e-70"):
        disc = mp.mpf(0)
    g_oa = (s_value - rho_value) ** 2 * disc**2 / (32 * a_value**7 * b_value**7 * rho_value**2)
    g_pmid = (s_value - t_value) ** 2 * disc / (8 * a_value**3 * b_value**5 * rho_value**2)
    return g_oa + g_pmid


def route_a_kernel_mp(k: int, n_shell: int, m_shell: int) -> mp.mpf:
    """Closed-form Route A joint kernel K(s,t) at high precision."""
    scale = mp.mpf(2) ** k
    s = mp.mpf(n_shell) / scale
    t = mp.mpf(m_shell) / scale
    a_value = mp.sqrt(s)
    b_value = mp.sqrt(t)
    log2 = mp.log(2)
    p_oa = (
        5 * s**6 - 20 * s**5 * t + 30 * s**4 * t**2 + 150 * s**4
        - 20 * s**3 * t**3 - 80 * s**3 * t - 300 * s**3
        + 5 * s**2 * t**4 + 40 * s**2 * t**2 - 120 * s**2 * t + 350 * s**2
        + 40 * s * t**3 - 120 * s * t**2 + 280 * s * t
        - 20 * s * (s - t) ** 2 * (3 * s**2 + t**2) * log2
        - 225 * s + 10 * t**4 - 60 * t**3 + 140 * t**2 - 150 * t + 62
    )
    m_oa = p_oa / (320 * a_value**7 * b_value**7)
    bracket = -1 + 2 * (s + t) * log2 + mp.mpf("0.5") * (4 * s * t - (s + t) ** 2)
    m_pmid = (s - t) ** 2 / (8 * a_value**3 * b_value**5) * bracket
    return m_oa + m_pmid


def route_a_integrand_iv(k: int, n_shell: int, m_shell: int, ell_shell: int) -> mp.iv.mpf:
    """Interval Route A joint integrand g(s,t,rho)."""
    scale = mp.iv.mpf([2**k, 2**k])
    s_value = mp.iv.mpf([n_shell, n_shell]) / scale
    t_value = mp.iv.mpf([m_shell, m_shell]) / scale
    rho_value = mp.iv.mpf([ell_shell, ell_shell]) / scale
    a_value = mp.iv.sqrt(s_value)
    b_value = mp.iv.sqrt(t_value)
    disc = 4 * s_value * t_value - (rho_value - s_value - t_value) ** 2
    g_oa = (s_value - rho_value) ** 2 * disc**2 / (32 * a_value**7 * b_value**7 * rho_value**2)
    g_pmid = (s_value - t_value) ** 2 * disc / (8 * a_value**3 * b_value**5 * rho_value**2)
    return g_oa + g_pmid


def route_a_kernel_iv(k: int, n_shell: int, m_shell: int) -> mp.iv.mpf:
    """Interval closed-form Route A joint kernel K(s,t)."""
    scale = mp.iv.mpf([2**k, 2**k])
    s = mp.iv.mpf([n_shell, n_shell]) / scale
    t = mp.iv.mpf([m_shell, m_shell]) / scale
    a_value = mp.iv.sqrt(s)
    b_value = mp.iv.sqrt(t)
    log2 = mp.iv.log(mp.iv.mpf([2, 2]))
    p_oa = (
        5 * s**6 - 20 * s**5 * t + 30 * s**4 * t**2 + 150 * s**4
        - 20 * s**3 * t**3 - 80 * s**3 * t - 300 * s**3
        + 5 * s**2 * t**4 + 40 * s**2 * t**2 - 120 * s**2 * t + 350 * s**2
        + 40 * s * t**3 - 120 * s * t**2 + 280 * s * t
        - 20 * s * (s - t) ** 2 * (3 * s**2 + t**2) * log2
        - 225 * s + 10 * t**4 - 60 * t**3 + 140 * t**2 - 150 * t + 62
    )
    m_oa = p_oa / (320 * a_value**7 * b_value**7)
    bracket = -1 + 2 * (s + t) * log2 + mp.iv.mpf(["0.5", "0.5"]) * (4 * s * t - (s + t) ** 2)
    m_pmid = (s - t) ** 2 / (8 * a_value**3 * b_value**5) * bracket
    return m_oa + m_pmid


def interval_lower(value: mp.iv.mpf) -> mp.mpf:
    return mp.mpf(value.a)


def interval_upper(value: mp.iv.mpf) -> mp.mpf:
    return mp.mpf(value.b)


def prepare_block(shells: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mode_blocks = []
    shell_values = []
    shell_indices = []
    for shell_index, shell in enumerate(shells):
        modes = lattice_shell(shell)
        mode_blocks.append(modes)
        shell_values.append(np.full(len(modes), shell, dtype=np.int64))
        shell_indices.append(np.full(len(modes), shell_index, dtype=np.int32))
    return np.vstack(mode_blocks), np.concatenate(shell_values), np.concatenate(shell_indices)


def cubic_orbit_representatives(shell: int) -> list[tuple[np.ndarray, int]]:
    """Return one representative per signed-permutation orbit on a shell."""
    orbit_sizes: Counter[tuple[int, int, int]] = Counter()
    for mode in lattice_shell(shell):
        key = tuple(sorted(abs(int(coord)) for coord in mode))
        orbit_sizes[key] += 1
    return [(np.array(key, dtype=np.int64), size) for key, size in sorted(orbit_sizes.items())]


def row_counts(
    k: int,
    n_shell: int,
    shells: list[int],
    block_modes: np.ndarray,
    block_shell_values: np.ndarray,
    block_shell_indices: np.ndarray,
    progress_every: int,
) -> Counter[tuple[int, int]]:
    """Return exact counts keyed by `(m_shell, ell_shell)` for one row."""
    scale = 2**k
    p_orbits = cubic_orbit_representatives(n_shell)
    counts: Counter[tuple[int, int]] = Counter()
    code_modulus = 2 * scale

    for p_index, (p_mode, orbit_size) in enumerate(p_orbits, start=1):
        dots = block_modes @ p_mode
        ell_values = n_shell + block_shell_values + 2 * dots
        mask = (ell_values >= scale) & (ell_values < 2 * scale)
        mask &= ~((block_shell_values == n_shell) & (ell_values == n_shell))
        if np.any(mask):
            codes = block_shell_indices[mask].astype(np.int64) * code_modulus + ell_values[mask]
            unique_codes, unique_counts = np.unique(codes, return_counts=True)
            for code, count in zip(unique_codes, unique_counts):
                shell_index = int(code // code_modulus)
                ell_shell = int(code % code_modulus)
                counts[(shells[shell_index], ell_shell)] += int(count) * orbit_size
        if progress_every > 0 and p_index % progress_every == 0:
            print(f"      counted n={n_shell} orbits={p_index}/{len(p_orbits)} cells={len(counts)}", flush=True)

    return counts


def verify_row(
    k: int,
    n_shell: int,
    shells: list[int],
    shell_counts: dict[int, int],
    block_modes: np.ndarray,
    block_shell_values: np.ndarray,
    block_shell_indices: np.ndarray,
    budget_ratio: mp.mpf,
    progress_every: int,
) -> None:
    counts = row_counts(k, n_shell, shells, block_modes, block_shell_values, block_shell_indices, progress_every)
    actual = mp.mpf(0)
    actual_by_m: defaultdict[int, mp.mpf] = defaultdict(mp.mpf)
    pair_total = 0
    for (m_shell, ell_shell), count in counts.items():
        weight = route_a_integrand_mp(k, n_shell, m_shell, ell_shell)
        contribution = count * weight
        actual += contribution
        actual_by_m[m_shell] += contribution
        pair_total += count

    scale = mp.mpf(2) ** k
    r3_n = shell_counts[n_shell]
    continuum = mp.mpf(0)
    worst_m = None
    worst_ratio = mp.mpf(-1)
    positive_excess = mp.mpf(0)
    for m_shell in shells:
        row_continuum = (
            r3_n
            * shell_counts[m_shell]
            * scale
            * route_a_kernel_mp(k, n_shell, m_shell)
            / (4 * mp.sqrt(n_shell * m_shell))
        )
        continuum += row_continuum
        row_actual = actual_by_m[m_shell]
        if row_actual > row_continuum:
            positive_excess += row_actual - row_continuum
        ratio = row_actual / row_continuum if row_continuum > 0 else mp.mpf(0)
        if ratio > worst_ratio:
            worst_ratio = ratio
            worst_m = m_shell

    ratio = actual / continuum
    print(
        f"n={n_shell} r3={r3_n} cells={len(counts)} pairs={pair_total} "
        f"actual={mp.nstr(actual, 18)} continuum={mp.nstr(continuum, 18)} "
        f"ratio={mp.nstr(ratio, 16)} pos_excess={mp.nstr(positive_excess, 12)} "
        f"worst_m={worst_m} worst_ratio={mp.nstr(worst_ratio, 12)} "
        f"budget_margin={mp.nstr(budget_ratio - ratio, 12)}",
        flush=True,
    )
    if ratio >= budget_ratio:
        raise SystemExit(f"row {n_shell} exceeds budget ratio {budget_ratio}")


def verify_row_interval(
    k: int,
    n_shell: int,
    shells: list[int],
    shell_counts: dict[int, int],
    block_modes: np.ndarray,
    block_shell_values: np.ndarray,
    block_shell_indices: np.ndarray,
    budget_ratio: mp.mpf,
    progress_every: int,
) -> None:
    counts = row_counts(k, n_shell, shells, block_modes, block_shell_values, block_shell_indices, progress_every)
    actual = mp.iv.mpf([0, 0])
    for (m_shell, ell_shell), count in counts.items():
        actual += count * route_a_integrand_iv(k, n_shell, m_shell, ell_shell)

    scale = mp.iv.mpf([2**k, 2**k])
    r3_n = shell_counts[n_shell]
    continuum = mp.iv.mpf([0, 0])
    for m_shell in shells:
        continuum += (
            r3_n
            * shell_counts[m_shell]
            * scale
            * route_a_kernel_iv(k, n_shell, m_shell)
            / (4 * mp.iv.sqrt(mp.iv.mpf([n_shell * m_shell, n_shell * m_shell])))
        )

    continuum_lower = interval_lower(continuum)
    if continuum_lower <= 0:
        raise SystemExit(f"row {n_shell} has nonpositive continuum interval lower endpoint: {continuum}")
    ratio_upper = interval_upper(actual) / continuum_lower
    print(
        f"n={n_shell} r3={r3_n} cells={len(counts)} "
        f"actual_upper={mp.nstr(interval_upper(actual), 18)} "
        f"continuum_lower={mp.nstr(continuum_lower, 18)} "
        f"ratio_upper={mp.nstr(ratio_upper, 16)} "
        f"budget_margin={mp.nstr(budget_ratio - ratio_upper, 12)}",
        flush=True,
    )
    if ratio_upper >= budget_ratio:
        raise SystemExit(f"row {n_shell} interval upper ratio exceeds budget ratio {budget_ratio}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=11)
    parser.add_argument("--shell", action="append", type=int, default=[], help="explicit first shell row to verify")
    parser.add_argument("--max-r3", type=int, help="verify all rows with r3(n) <= this value")
    parser.add_argument("--min-r3", type=int, default=0, help="when using --max-r3, require r3(n) >= this value")
    parser.add_argument("--all", action="store_true", help="verify every represented row in the k-block")
    parser.add_argument("--dps", type=int, default=80)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--budget-ratio", default="1.1503561001915947")
    parser.add_argument("--interval", action="store_true", help="use mpmath interval arithmetic for the final scalar weights")
    parser.add_argument("--workers", type=int, default=0, help="row-level worker threads; 0 uses all logical CPUs")
    args = parser.parse_args()

    mp.mp.dps = args.dps
    mp.iv.dps = args.dps
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

    row_description = [(shell, shell_counts[shell]) for shell in rows] if len(rows) <= 20 else f"{len(rows)} represented rows"
    print(f"k={args.k}, dps={args.dps}, rows={row_description}")
    print(f"budget ratio={args.budget_ratio}")
    block_modes, block_shell_values, block_shell_indices = prepare_block(shells)
    budget_ratio = mp.mpf(args.budget_ratio)
    worker_count = args.workers if args.workers > 0 else default_worker_count(len(rows))
    print(f"row workers={worker_count}")
    verifier = verify_row_interval if args.interval else verify_row

    def run_row(n_shell: int) -> None:
        verifier(
            args.k,
            n_shell,
            shells,
            shell_counts,
            block_modes,
            block_shell_values,
            block_shell_indices,
            budget_ratio,
            args.progress_every if worker_count == 1 else 0,
        )

    if worker_count == 1:
        for n_shell in rows:
            run_row(n_shell)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(run_row, n_shell) for n_shell in rows]
            for future in as_completed(futures):
                future.result()


if __name__ == "__main__":
    main()