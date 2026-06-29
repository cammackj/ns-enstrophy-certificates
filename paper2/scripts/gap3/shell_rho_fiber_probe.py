#!/usr/bin/env python3
"""Probe discrete angular fibers for the k11 shell/rho finite-sum route.

For shells n=|p|^2, m=|q|^2, l=|p+q|^2, the angular fiber over fixed p is

    {q in Z^3 : |q|^2=m, p.q=(l-n-m)/2}.

The Route A radial finite-sum target requires a discrete angular/fiber
domination theorem.  This script does not prove such a theorem; it measures the
arithmetic concentration that the theorem must handle.
"""

from __future__ import annotations

import argparse
import math
from functools import lru_cache
from typing import Iterable

import numpy as np


@lru_cache(maxsize=None)
def lattice_shell(shell: int) -> np.ndarray:
    """Return all integer vectors with squared length shell."""
    limit = math.isqrt(shell)
    modes: list[tuple[int, int, int]] = []
    for x in range(-limit, limit + 1):
        rem_yz = shell - x * x
        if rem_yz < 0:
            continue
        limit_y = math.isqrt(rem_yz)
        for y in range(-limit_y, limit_y + 1):
            z2 = rem_yz - y * y
            z = math.isqrt(z2)
            if z * z != z2:
                continue
            if z == 0:
                modes.append((x, y, 0))
            else:
                modes.append((x, y, z))
                modes.append((x, y, -z))
    return np.asarray(modes, dtype=np.int64)


def represented_shells(k: int) -> list[int]:
    return [shell for shell in range(2**k, 2 ** (k + 1)) if len(lattice_shell(shell)) > 0]


def fiber_stats(n: int, m: int, ell_shell: int) -> dict[str, float | int]:
    """Count angular fibers for p on shell n, q on shell m, |p+q|^2=ell_shell."""
    parity = ell_shell - n - m
    if parity % 2 != 0:
        return {
            "n": n,
            "m": m,
            "ell": ell_shell,
            "target_dot": 0,
            "r3_n": len(lattice_shell(n)),
            "r3_m": len(lattice_shell(m)),
            "active_p": 0,
            "total_pairs": 0,
            "max_fiber": 0,
            "mean_active": 0.0,
            "continuum_cell": 0.0,
            "max_over_continuum": 0.0,
            "total_over_continuum": 0.0,
        }
    target_dot = parity // 2
    p_modes = lattice_shell(n)
    q_modes = lattice_shell(m)
    if len(p_modes) == 0 or len(q_modes) == 0:
        raise ValueError("empty represented shell passed to fiber_stats")
    dots = p_modes @ q_modes.T
    counts = np.count_nonzero(dots == target_dot, axis=1)
    total_pairs = int(counts.sum())
    active = counts[counts > 0]
    continuum_cell = len(q_modes) / (2.0 * math.sqrt(n * m))
    total_continuum = len(p_modes) * continuum_cell
    return {
        "n": n,
        "m": m,
        "ell": ell_shell,
        "target_dot": int(target_dot),
        "r3_n": int(len(p_modes)),
        "r3_m": int(len(q_modes)),
        "active_p": int(len(active)),
        "total_pairs": total_pairs,
        "max_fiber": int(counts.max(initial=0)),
        "mean_active": float(active.mean()) if len(active) else 0.0,
        "continuum_cell": float(continuum_cell),
        "max_over_continuum": float(counts.max(initial=0) / continuum_cell) if continuum_cell > 0 else 0.0,
        "total_over_continuum": float(total_pairs / total_continuum) if total_continuum > 0 else 0.0,
    }


def candidate_triples(k: int, stride: int, limit: int) -> Iterable[tuple[int, int, int]]:
    shells = represented_shells(k)
    picked = shells[::stride]
    emitted = 0
    for n in picked:
        for m in picked:
            lo = max(2**k, int((math.sqrt(n) - math.sqrt(m)) ** 2) - 2)
            hi = min(2 ** (k + 1) - 1, int((math.sqrt(n) + math.sqrt(m)) ** 2) + 2)
            for ell_shell in range(lo, hi + 1, stride):
                if (ell_shell - n - m) % 2 != 0:
                    continue
                if len(lattice_shell(ell_shell)) == 0:
                    continue
                if n == m == ell_shell:
                    continue
                yield n, m, ell_shell
                emitted += 1
                if emitted >= limit:
                    return


def print_row(stats: dict[str, float | int]) -> None:
    print(
        f"{stats['n']:5d} {stats['m']:5d} {stats['ell']:5d} {stats['target_dot']:7d} "
        f"{stats['r3_n']:5d} {stats['r3_m']:5d} {stats['active_p']:5d} "
        f"{stats['total_pairs']:7d} {stats['max_fiber']:5d} "
        f"{stats['mean_active']:9.3f} {stats['continuum_cell']:10.4f} "
        f"{stats['max_over_continuum']:10.3f} {stats['total_over_continuum']:10.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=11)
    parser.add_argument("--stride", type=int, default=29, help="shell stride for automatic sampling")
    parser.add_argument("--limit", type=int, default=200, help="maximum sampled triples")
    parser.add_argument("--include-zero", action="store_true", help="include triples with no lattice pairs")
    parser.add_argument("--include-same-shell", action="store_true", help="include n=m=ell triples deleted by same-shell vanishing")
    parser.add_argument(
        "--triple",
        action="append",
        default=[],
        metavar="N,M,L",
        help="explicit shell triple to inspect; may be repeated",
    )
    args = parser.parse_args()

    triples: list[tuple[int, int, int]] = []
    for item in args.triple:
        pieces = [int(part) for part in item.split(",")]
        if len(pieces) != 3:
            raise ValueError(f"expected N,M,L triple, got {item!r}")
        triples.append(tuple(pieces))
    if not triples:
        triples = list(candidate_triples(args.k, args.stride, args.limit))

    if not args.include_same_shell:
        triples = [triple for triple in triples if not (triple[0] == triple[1] == triple[2])]
    rows = [fiber_stats(*triple) for triple in triples]
    if not args.include_zero:
        rows = [row for row in rows if int(row["total_pairs"]) > 0]
    rows.sort(key=lambda row: (float(row["max_over_continuum"]), float(row["total_over_continuum"])), reverse=True)

    print(f"k={args.k}, inspected triples={len(rows)}")
    print(
        "    n     m   ell     dot  r3_n  r3_m active   pairs   max "
        "mean_act  cont_cell   max/cont total/cont"
    )
    for row in rows[: min(len(rows), 40)]:
        print_row(row)

    if rows:
        worst = rows[0]
        print("\nworst sampled max/continuum ratio:")
        print_row(worst)


if __name__ == "__main__":
    main()