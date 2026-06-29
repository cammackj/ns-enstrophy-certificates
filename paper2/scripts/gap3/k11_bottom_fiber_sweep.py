#!/usr/bin/env python3
"""Sweep angular fibers involving the k=11 bottom D shell.

The k11 shell/rho route cannot rely on naive angular equidistribution because
low-representation shells have concentrated fibers.  This diagnostic fixes
n=2048 and counts all pairs (p,q) with p on the bottom D shell and q in I_11,
grouped by (m=|q|^2, ell=|p+q|^2).
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict

import numpy as np

from shell_rho_fiber_probe import lattice_shell, represented_shells


LEGENDRE_X, LEGENDRE_W = np.polynomial.legendre.leggauss(96)


def route_a_kernel(s_value: float, t_value: float) -> float:
    """Closed-form Route A joint kernel K(s,t)."""
    a_value = math.sqrt(s_value)
    b_value = math.sqrt(t_value)
    log2 = math.log(2.0)
    s = s_value
    t = t_value
    p_oa = (
        5 * s**6 - 20 * s**5 * t + 30 * s**4 * t**2 + 150 * s**4
        - 20 * s**3 * t**3 - 80 * s**3 * t - 300 * s**3
        + 5 * s**2 * t**4 + 40 * s**2 * t**2 - 120 * s**2 * t + 350 * s**2
        + 40 * s * t**3 - 120 * s * t**2 + 280 * s * t
        - 20 * s * (s - t) ** 2 * (3 * s**2 + t**2) * log2
        - 225 * s + 10 * t**4 - 60 * t**3 + 140 * t**2 - 150 * t + 62
    )
    m_oa = p_oa / (320.0 * a_value**7 * b_value**7)
    bracket = -1.0 + 2.0 * (s + t) * log2 + 0.5 * (4.0 * s * t - (s + t) ** 2)
    m_pmid = (s - t) ** 2 / (8.0 * a_value**3 * b_value**5) * bracket
    return m_oa + m_pmid


def route_a_integrand(s_value: float, t_value: float, rho_value: np.ndarray | float) -> np.ndarray | float:
    """Numerical joint Route A integrand before rho integration."""
    a_value = math.sqrt(s_value)
    b_value = math.sqrt(t_value)
    rho = np.asarray(rho_value, dtype=np.float64)
    disc = 4.0 * s_value * t_value - (rho - s_value - t_value) ** 2
    disc = np.maximum(disc, 0.0)
    g_oa = (s_value - rho) ** 2 * disc**2 / (32.0 * a_value**7 * b_value**7 * rho**2)
    g_pmid = (s_value - t_value) ** 2 * disc / (8.0 * a_value**3 * b_value**5 * rho**2)
    return g_oa + g_pmid


def route_a_integral(s_value: float, t_value: float) -> float:
    """Gauss-Legendre diagnostic integral of the joint Route A integrand on rho in [1,2]."""
    rho = 1.5 + 0.5 * LEGENDRE_X
    return float(0.5 * np.dot(LEGENDRE_W, route_a_integrand(s_value, t_value, rho)))


def sweep_bottom(k: int, bottom_shell: int) -> list[dict[str, float | int]]:
    scale = 2**k
    lo = 2**k
    hi = 2 ** (k + 1)
    p_modes = lattice_shell(bottom_shell)
    rows: dict[tuple[int, int], dict[str, object]] = {}

    for m in represented_shells(k):
        q_modes = lattice_shell(m)
        s_value = bottom_shell / scale
        t_value = m / scale
        k_integral = route_a_kernel(s_value, t_value)
        continuum_g_sum = (
            len(p_modes)
            * len(q_modes)
            * scale
            * k_integral
            / (4.0 * math.sqrt(bottom_shell * m))
        )
        per_ell_counts: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(len(p_modes), dtype=np.int64))
        for p_index, p in enumerate(p_modes):
            sums = q_modes + p
            ell_values = np.einsum("ij,ij->i", sums, sums)
            mask = (ell_values >= lo) & (ell_values < hi)
            if not np.any(mask):
                continue
            unique, counts = np.unique(ell_values[mask], return_counts=True)
            for ell_shell, count in zip(unique, counts):
                if bottom_shell == m == int(ell_shell):
                    continue
                per_ell_counts[int(ell_shell)][p_index] += int(count)

        continuum_cell = len(q_modes) / (2.0 * math.sqrt(bottom_shell * m))
        total_continuum = len(p_modes) * continuum_cell
        actual_g_total = 0.0
        for ell_shell, counts in per_ell_counts.items():
            actual_g_total += float(counts.sum()) * float(route_a_integrand(s_value, t_value, ell_shell / scale))
        for ell_shell, counts in per_ell_counts.items():
            total_pairs = int(counts.sum())
            if total_pairs == 0:
                continue
            g_value = float(route_a_integrand(s_value, t_value, ell_shell / scale))
            active = counts[counts > 0]
            rows[(m, ell_shell)] = {
                "n": bottom_shell,
                "m": m,
                "ell": ell_shell,
                "r3_m": int(len(q_modes)),
                "active_p": int(len(active)),
                "total_pairs": total_pairs,
                "max_fiber": int(counts.max(initial=0)),
                "mean_active": float(active.mean()) if len(active) else 0.0,
                "continuum_cell": float(continuum_cell),
                "max_over_continuum": float(counts.max(initial=0) / continuum_cell) if continuum_cell > 0 else 0.0,
                "total_over_continuum": float(total_pairs / total_continuum) if total_continuum > 0 else 0.0,
                "g_value": g_value,
                "weighted_mass": float(total_pairs * g_value),
                "m_weighted_ratio": float(actual_g_total / continuum_g_sum) if continuum_g_sum > 0 else 0.0,
            }

    return list(rows.values())


def print_rows(rows: list[dict[str, float | int]], limit: int) -> None:
    print(
        "    n     m   ell  r3_m active   pairs   max mean_act  "
        "cont_cell   max/cont total/cont       g     wg_mass  m_wg/cont"
    )
    for row in rows[:limit]:
        print(
            f"{row['n']:5d} {row['m']:5d} {row['ell']:5d} {row['r3_m']:5d} "
            f"{row['active_p']:5d} {row['total_pairs']:7d} {row['max_fiber']:5d} "
            f"{row['mean_active']:9.3f} {row['continuum_cell']:10.4f} "
            f"{row['max_over_continuum']:10.3f} {row['total_over_continuum']:10.3f} "
            f"{row['g_value']:7.4f} {row['weighted_mass']:11.4f} {row['m_weighted_ratio']:10.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=11)
    parser.add_argument("--bottom-shell", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    rows = sweep_bottom(args.k, args.bottom_shell)
    print(f"k={args.k}, bottom_shell={args.bottom_shell}, nonzero shell-pair fibers={len(rows)}")
    print(f"total bottom-involving pairs: {sum(int(row['total_pairs']) for row in rows):,}")
    print(f"maximum single-p fiber count in one shell/rho fiber: {max((int(row['max_fiber']) for row in rows), default=0)}")
    print(f"maximum total pairs in one shell/rho fiber: {max((int(row['total_pairs']) for row in rows), default=0)}")
    actual_by_m: dict[int, float] = defaultdict(float)
    ratio_by_m: dict[int, float] = {}
    for row in rows:
        actual_by_m[int(row["m"])] += float(row["weighted_mass"])
        ratio_by_m[int(row["m"])] = float(row["m_weighted_ratio"])
    continuum_total = sum(
        actual / ratio_by_m[m]
        for m, actual in actual_by_m.items()
        if ratio_by_m[m] > 0
    )
    actual_total = sum(actual_by_m.values())
    print(f"Route A weighted actual sum: {actual_total:.6f}")
    print(f"Route A weighted continuum estimate: {continuum_total:.6f}")
    print(f"Route A weighted aggregate excess ratio: {actual_total / continuum_total:.6f}")

    print("\nTop by max/continuum one-cell ratio")
    by_ratio = sorted(rows, key=lambda row: (float(row["max_over_continuum"]), float(row["total_over_continuum"])), reverse=True)
    print_rows(by_ratio, args.limit)

    print("\nTop by total pair count")
    by_pairs = sorted(rows, key=lambda row: int(row["total_pairs"]), reverse=True)
    print_rows(by_pairs, args.limit)

    print("\nTop by Route A weighted shell/rho mass")
    by_weight = sorted(rows, key=lambda row: float(row["weighted_mass"]), reverse=True)
    print_rows(by_weight, args.limit)

    by_m_weight = sorted(rows, key=lambda row: float(row["m_weighted_ratio"]), reverse=True)
    seen_m = set()
    worst_m_rows = []
    for row in by_m_weight:
        if row["m"] in seen_m:
            continue
        seen_m.add(row["m"])
        worst_m_rows.append(row)
    print("\nTop by full-m weighted excess ratio")
    print_rows(worst_m_rows, args.limit)


if __name__ == "__main__":
    main()