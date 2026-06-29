#!/usr/bin/env python3
"""Probe weighted shell-pair sums for the k11 Route A shell/rho lemma.

For an ordered shell pair (n,m), this computes the exact discrete sum of the
Route A joint integrand g(s,t,rho) over all pairs p in N_n, q in N_m whose
sum p+q lands in I_k, then compares it with the continuum coarea estimate

    r3(n) r3(m) 2^k /(4 sqrt(nm)) * integral_1^2 g(n/2^k,m/2^k,rho) drho.

This is a diagnostic for the weighted angular/fiber lemma.  It is not a proof
certificate: the integral is numerical and no interval arithmetic is used.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np

from k11_bottom_fiber_sweep import route_a_integrand, route_a_kernel
from shell_rho_fiber_probe import lattice_shell, represented_shells


@dataclass(frozen=True)
class PairStats:
    n: int
    m: int
    r3_n: int
    r3_m: int
    active_pairs: int
    actual: float
    continuum: float
    ratio: float
    max_fiber: int


def weighted_pair_stats(k: int, n: int, m: int) -> PairStats:
    scale = 2**k
    lo = scale
    hi = 2 * scale
    p_modes = lattice_shell(n)
    q_modes = lattice_shell(m)
    s_value = n / scale
    t_value = m / scale
    kernel = route_a_kernel(s_value, t_value)
    continuum = len(p_modes) * len(q_modes) * scale * kernel / (4.0 * math.sqrt(n * m))

    actual = 0.0
    active_pairs = 0
    max_fiber = 0
    for p in p_modes:
        sums = q_modes + p
        ell_values = np.einsum("ij,ij->i", sums, sums)
        mask = (ell_values >= lo) & (ell_values < hi)
        if n == m:
            mask &= ell_values != n
        if not np.any(mask):
            continue
        active = ell_values[mask]
        active_pairs += int(active.size)
        unique, counts = np.unique(active, return_counts=True)
        max_fiber = max(max_fiber, int(counts.max(initial=0)))
        rho_values = unique.astype(np.float64) / scale
        actual += float(np.dot(counts.astype(np.float64), route_a_integrand(s_value, t_value, rho_values)))

    ratio = actual / continuum if continuum > 0 else 0.0
    return PairStats(n, m, len(p_modes), len(q_modes), active_pairs, actual, continuum, ratio, max_fiber)


def parse_pair_list(raw: str) -> list[tuple[int, int]]:
    pairs = []
    for item in raw.replace(";", ":").split(":"):
        item = item.strip()
        if not item:
            continue
        left, right = item.split(",")
        pairs.append((int(left), int(right)))
    return pairs


def sampled_pairs(k: int, stride: int, limit: int, max_product: int) -> list[tuple[int, int]]:
    shells = represented_shells(k)
    grid = shells[::stride]
    pairs = []
    for n in grid:
        r3_n = len(lattice_shell(n))
        for m in grid:
            r3_m = len(lattice_shell(m))
            if r3_n * r3_m > max_product:
                continue
            pairs.append((n, m))
            if len(pairs) >= limit:
                return pairs
    return pairs


def print_stats(rows: list[PairStats], limit: int) -> None:
    print("    n     m  r3_n  r3_m    pairs maxfib       actual    continuum    ratio")
    for row in rows[:limit]:
        print(
            f"{row.n:5d} {row.m:5d} {row.r3_n:5d} {row.r3_m:5d} "
            f"{row.active_pairs:8d} {row.max_fiber:6d} "
            f"{row.actual:12.6f} {row.continuum:12.6f} {row.ratio:8.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=11)
    parser.add_argument("--pairs", help="explicit ordered pairs, e.g. 2048,2483:3088,2064")
    parser.add_argument("--stride", type=int, default=73)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--max-product", type=int, default=2_000_000)
    args = parser.parse_args()

    pairs = parse_pair_list(args.pairs) if args.pairs else sampled_pairs(args.k, args.stride, args.limit, args.max_product)
    rows = [weighted_pair_stats(args.k, n, m) for n, m in pairs]
    rows = [row for row in rows if row.continuum > 0]
    print(f"k={args.k}, inspected ordered pairs={len(rows)}")

    print("\nTop by weighted excess ratio")
    print_stats(sorted(rows, key=lambda row: row.ratio, reverse=True), args.limit)

    print("\nTop by exact weighted mass")
    print_stats(sorted(rows, key=lambda row: row.actual, reverse=True), args.limit)


if __name__ == "__main__":
    main()