#!/usr/bin/env python3
"""High-precision row verifier for the midpoint-centered raw-output kernel.

The verifier counts exact integer `(m, ell)` shell/rho cells for a fixed row and
then evaluates the centered scalar weights.  With `--interval`, the final scalar
weights use mpmath interval arithmetic and the reported ratio is
actual_upper/continuum_lower.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import mpmath as mp
import numpy as np

from shell_rho_fiber_probe import lattice_shell, represented_shells


def default_worker_count(row_count: int) -> int:
    return max(1, min(os.cpu_count() or 1, row_count))


def poly_mul(left, right):
    zero = left[0] * 0
    out = [zero] * (len(left) + len(right) - 1)
    for i, left_value in enumerate(left):
        for j, right_value in enumerate(right):
            out[i + j] += left_value * right_value
    return out


def centered_integrand_mp(k: int, n_shell: int, m_shell: int, ell_shell: int) -> mp.mpf:
    scale = mp.mpf(2) ** k
    s_value = mp.mpf(n_shell) / scale
    t_value = mp.mpf(m_shell) / scale
    rho_value = mp.mpf(ell_shell) / scale
    disc = 4 * s_value * t_value - (rho_value - s_value - t_value) ** 2
    st_value = s_value * t_value
    return (rho_value - mp.mpf("1.5")) ** 2 * disc**2 / (32 * st_value**3 * mp.sqrt(st_value) * rho_value**2)


def centered_integrand_iv(k: int, n_shell: int, m_shell: int, ell_shell: int) -> mp.iv.mpf:
    scale = mp.iv.mpf([2**k, 2**k])
    s_value = mp.iv.mpf([n_shell, n_shell]) / scale
    t_value = mp.iv.mpf([m_shell, m_shell]) / scale
    rho_value = mp.iv.mpf([ell_shell, ell_shell]) / scale
    disc = 4 * s_value * t_value - (rho_value - s_value - t_value) ** 2
    st_value = s_value * t_value
    return (rho_value - mp.iv.mpf(["1.5", "1.5"])) ** 2 * disc**2 / (32 * st_value**3 * mp.iv.sqrt(st_value) * rho_value**2)


def centered_kernel_mp(k: int, n_shell: int, m_shell: int) -> mp.mpf:
    """Closed-form integral of the centered rho-kernel from rho=1 to rho=2."""
    scale = mp.mpf(2) ** k
    s_value = mp.mpf(n_shell) / scale
    t_value = mp.mpf(m_shell) / scale
    u_value = s_value + t_value
    constant = 4 * s_value * t_value - u_value**2
    # disc = -rho^2 + 2(s+t)rho + constant.
    disc = [constant, 2 * u_value, mp.mpf(-1)]
    centered = [mp.mpf("2.25"), mp.mpf(-3), mp.mpf(1)]
    numerator = poly_mul(centered, poly_mul(disc, disc))
    integral = mp.mpf(0)
    for power, coeff in enumerate(numerator):
        exponent = power - 2
        if exponent == -2:
            integral += coeff * mp.mpf("0.5")
        elif exponent == -1:
            integral += coeff * mp.log(2)
        else:
            integral += coeff * (mp.power(2, exponent + 1) - 1) / (exponent + 1)
    st_value = s_value * t_value
    return integral / (32 * st_value**3 * mp.sqrt(st_value))


def centered_kernel_iv(k: int, n_shell: int, m_shell: int) -> mp.iv.mpf:
    """Interval enclosure for the closed-form centered rho-kernel."""
    scale = mp.iv.mpf([2**k, 2**k])
    s_value = mp.iv.mpf([n_shell, n_shell]) / scale
    t_value = mp.iv.mpf([m_shell, m_shell]) / scale
    u_value = s_value + t_value
    constant = 4 * s_value * t_value - u_value**2
    disc = [constant, 2 * u_value, mp.iv.mpf([-1, -1])]
    centered = [mp.iv.mpf(["2.25", "2.25"]), mp.iv.mpf([-3, -3]), mp.iv.mpf([1, 1])]
    numerator = poly_mul(centered, poly_mul(disc, disc))
    integral = mp.iv.mpf([0, 0])
    two = mp.iv.mpf([2, 2])
    for power, coeff in enumerate(numerator):
        exponent = power - 2
        if exponent == -2:
            integral += coeff * mp.iv.mpf(["0.5", "0.5"])
        elif exponent == -1:
            integral += coeff * mp.iv.log(two)
        else:
            integral += coeff * (two ** (exponent + 1) - 1) / (exponent + 1)
    st_value = s_value * t_value
    return integral / (32 * st_value**3 * mp.iv.sqrt(st_value))


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
    scale = 2**k
    counts: Counter[tuple[int, int]] = Counter()
    code_modulus = 2 * scale
    for orbit_index, (p_mode, orbit_size) in enumerate(cubic_orbit_representatives(n_shell), start=1):
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
        if progress_every > 0 and orbit_index % progress_every == 0:
            print(f"      counted n={n_shell} orbits={orbit_index} cells={len(counts)}", flush=True)
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
) -> tuple[mp.mpf, int, int, int, mp.mpf, mp.mpf]:
    counts = row_counts(k, n_shell, shells, block_modes, block_shell_values, block_shell_indices, progress_every)
    actual = mp.mpf(0)
    pair_total = 0
    for (m_shell, ell_shell), count in counts.items():
        actual += count * centered_integrand_mp(k, n_shell, m_shell, ell_shell)
        pair_total += count

    scale = mp.mpf(2) ** k
    continuum = mp.mpf(0)
    r3_n = shell_counts[n_shell]
    for m_shell in shells:
        continuum += (
            r3_n
            * shell_counts[m_shell]
            * scale
            * centered_kernel_mp(k, n_shell, m_shell)
            / (4 * mp.sqrt(n_shell * m_shell))
        )
    ratio = actual / continuum if continuum > 0 else mp.inf
    print(
        f"n={n_shell} r3={r3_n} cells={len(counts)} pairs={pair_total} "
        f"ratio={mp.nstr(ratio, 16)} actual={mp.nstr(actual, 16)} continuum={mp.nstr(continuum, 16)} "
        f"budget_margin={mp.nstr(budget_ratio - ratio, 12)}",
        flush=True,
    )
    if ratio >= budget_ratio:
        raise SystemExit(f"row {n_shell} exceeds budget ratio {budget_ratio}: {ratio}")
    return ratio, n_shell, r3_n, pair_total, actual, continuum


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
) -> tuple[mp.mpf, int, int, int, mp.mpf, mp.mpf]:
    counts = row_counts(k, n_shell, shells, block_modes, block_shell_values, block_shell_indices, progress_every)
    actual = mp.iv.mpf([0, 0])
    pair_total = 0
    for (m_shell, ell_shell), count in counts.items():
        actual += count * centered_integrand_iv(k, n_shell, m_shell, ell_shell)
        pair_total += count

    scale = mp.iv.mpf([2**k, 2**k])
    continuum = mp.iv.mpf([0, 0])
    r3_n = shell_counts[n_shell]
    for m_shell in shells:
        continuum += (
            r3_n
            * shell_counts[m_shell]
            * scale
            * centered_kernel_iv(k, n_shell, m_shell)
            / (4 * mp.iv.sqrt(mp.iv.mpf([n_shell * m_shell, n_shell * m_shell])))
        )
    continuum_lower = interval_lower(continuum)
    if continuum_lower <= 0:
        raise SystemExit(f"row {n_shell} has nonpositive continuum lower endpoint: {continuum}")
    actual_upper = interval_upper(actual)
    ratio_upper = actual_upper / continuum_lower
    print(
        f"n={n_shell} r3={r3_n} cells={len(counts)} pairs={pair_total} "
        f"ratio_upper={mp.nstr(ratio_upper, 16)} actual_upper={mp.nstr(actual_upper, 16)} "
        f"continuum_lower={mp.nstr(continuum_lower, 16)} "
        f"budget_margin={mp.nstr(budget_ratio - ratio_upper, 12)}",
        flush=True,
    )
    if ratio_upper >= budget_ratio:
        raise SystemExit(f"row {n_shell} interval upper ratio exceeds budget ratio {budget_ratio}: {ratio_upper}")
    return ratio_upper, n_shell, r3_n, pair_total, actual_upper, continuum_lower


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--shell", action="append", type=int, default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--max-r3", type=int)
    parser.add_argument("--min-r3", type=int, default=0)
    parser.add_argument("--dps", type=int, default=80)
    parser.add_argument("--budget-ratio", default="1.25")
    parser.add_argument("--interval", action="store_true", help="use mpmath interval arithmetic for the final scalar weights")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--top", type=int, default=20)
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
        rows = [shells[0]]
    missing = [shell for shell in rows if shell not in shell_counts]
    if missing:
        raise ValueError(f"requested shells are not represented in I_{args.k}: {missing}")

    print(f"k={args.k}, dps={args.dps}, rows={len(rows)}, budget={args.budget_ratio}")
    block_modes, block_shell_values, block_shell_indices = prepare_block(shells)
    budget_ratio = mp.mpf(args.budget_ratio)
    workers = args.workers if args.workers > 0 else default_worker_count(len(rows))
    print(f"row workers={workers}")

    verifier = verify_row_interval if args.interval else verify_row

    def run_row(shell: int) -> tuple[mp.mpf, int, int, int, mp.mpf, mp.mpf]:
        return verifier(
            args.k,
            shell,
            shells,
            shell_counts,
            block_modes,
            block_shell_values,
            block_shell_indices,
            budget_ratio,
            args.progress_every if workers == 1 else 0,
        )

    results = []
    if workers == 1:
        for shell in rows:
            results.append(run_row(shell))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_row, shell) for shell in rows]
            for future in as_completed(futures):
                results.append(future.result())

    print("\nTop ratios")
    for ratio, shell, r3, pair_total, actual, continuum in sorted(results, reverse=True)[: args.top]:
        print(
            f"n={shell} r3={r3} pairs={pair_total} ratio={mp.nstr(ratio, 16)} "
            f"margin={mp.nstr(budget_ratio - ratio, 12)}"
        )


if __name__ == "__main__":
    main()
