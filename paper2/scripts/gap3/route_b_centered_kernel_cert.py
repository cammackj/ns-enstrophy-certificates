#!/usr/bin/env python3
"""Audit the midpoint-centered Route B raw-output kernel.

This checks the scalar kernel obtained from the global identity
B(u,u,Delta u)=B(u,u,(Delta+mu)u), with mu equal to the midpoint of the active
squared-shell block after dyadic rescaling.  In annulus coordinates this replaces
the raw output coefficient rho by rho-3/2 and gives

    K_c(s,t) = int_1^2 (rho-3/2)^2 * disc^2
              / (32 (st)^(7/2) rho^2) d rho,
    disc = 4 s t - (rho-s-t)^2.

The script certifies conservative constants for K_c on a rational enlarged
square and prints the resulting radial finite-height budgets.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import sympy as sp

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gap3.route_a_joint_kernel_cert import (  # noqa: E402
    C2,
    LOG2_HI,
    LOG2_LO,
    verify_nonnegative_bernstein_linear_log2,
)


def centered_kernel_objects() -> tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
    """Return K_c(a,b), g_c(a,b,rho), a, b, rho, L."""
    a, b, rho, L = sp.symbols("a b rho L", positive=True)
    s = a * a
    t = b * b
    disc = 4 * s * t - (rho - s - t) ** 2
    g_center = sp.expand((rho - sp.Rational(3, 2)) ** 2 * disc**2 / (32 * a**7 * b**7 * rho**2))
    k_center = sp.integrate(g_center, (rho, sp.Integer(1), sp.Integer(2)))
    k_center = sp.expand(k_center.subs(sp.log(2), L))
    return k_center, g_center, a, b, rho, L


def verify_centered_constants() -> None:
    k_center, g_center, a, b, rho, L = centered_kernel_objects()
    x, y = sp.symbols("x y")
    a_lo = sp.Integer(1)
    a_hi = sp.Rational(10, 7)
    transform = {a: a_lo + (a_hi - a_lo) * x, b: a_lo + (a_hi - a_lo) * y}

    ds = sp.diff(k_center, a) / (2 * a)
    dt = sp.diff(k_center, b) / (2 * b)
    checks = [
        ("1/60 - K_center", sp.Rational(1, 60) - k_center),
        ("1/25 - partial_s K_center", sp.Rational(1, 25) - ds),
        ("1/25 + partial_s K_center", sp.Rational(1, 25) + ds),
        ("1/25 - partial_t K_center", sp.Rational(1, 25) - dt),
        ("1/25 + partial_t K_center", sp.Rational(1, 25) + dt),
        ("9/128 - collar rho=1", sp.Rational(9, 128) - g_center.subs(rho, 1)),
        ("9/128 - collar rho=2", sp.Rational(9, 128) - g_center.subs(rho, 2)),
    ]

    print("STEP 1  midpoint-centered kernel constants")
    print("        enlarged square: a,b in [1,10/7]")
    print("        target constants: K<=1/60, |partial_s K|,|partial_t K|<=1/25, collar<=9/128")
    for label, expr in checks:
        numerator = sp.together(expr).as_numer_denom()[0]
        poly = sp.expand(numerator.subs(transform))
        verify_nonnegative_bernstein_linear_log2(poly, x, y, L, label)
    print("        PASS: centered kernel constants certified on the enlarged square")


def radial_budget(k: int, derivative_bound: sp.Expr = sp.Rational(1, 25), collar_bound: sp.Expr = sp.Rational(9, 128)) -> tuple[sp.Expr, list[tuple[int, sp.Expr, sp.Expr, sp.Expr]]]:
    h = sp.Pow(2, -sp.Rational(k, 2))
    k_center = sp.Rational(1, 60)
    upper = sp.Integer(0)
    rows = []
    for j in range(0, (k - 1) // 2 + 1):
        alpha = sp.Integer(1) if j == 0 else sp.Integer(2) ** j
        weight = sp.Integer(1) if j == 0 else sp.Pow(2, -sp.Rational(3 * (j - 1), 2))
        perturbation = h * h * (derivative_bound + alpha * alpha * (derivative_bound + 2 * collar_bound))
        contribution = weight * sp.sqrt(k_center + perturbation)
        upper += contribution
        rows.append((j, weight, contribution, perturbation))
    return upper, rows


def print_budgets() -> None:
    print("\nSTEP 2  centered radial finite-height budgets")
    print("        uses K<=1/60, derivative<=1/25, collar<=9/128")
    for k in range(8, 13):
        upper, rows = radial_budget(k)
        threshold = C2 * sp.Pow(2, sp.Rational(k, 2))
        margin = threshold - upper
        status = "fits" if bool(sp.N(margin, 40) > 0) else "misses"
        print(
            f"        k={k:2d}  U_center,radial<={float(sp.N(upper, 30)):.12f}  "
            f"threshold={float(sp.N(threshold, 30)):.12f}  margin={float(sp.N(margin, 30)):+.12f}  {status}"
        )
        if k <= 10:
            for j, weight, contribution, perturbation in rows:
                print(
                    f"          j={j}: weight={float(sp.N(weight, 20)):.12f}  "
                    f"contrib={float(sp.N(contribution, 20)):.12f}  "
                    f"perturb={float(sp.N(perturbation, 20)):.12e}"
                )


CERTIFIED_LOW_RESCALED = {
    1: sp.Rational(0),
    2: C2 * 2,
    # Conservative displayed values from references/P11_CANCELLATION_BOUND.md.
    3: sp.Rational(62045600534, 10**12),
    4: sp.Rational(84258, 10**6),
    5: sp.Rational(104535, 10**6),
    6: sp.Rational(163550, 10**6),
    7: sp.Rational(229449, 10**6),
}


ROW_GUARD = {
    # k8/k6 are interval-checked by centered_row_mp_verify.py --interval.
    6: sp.Rational(5, 4),
    8: sp.Rational(5, 4),
    # k7 is interval-checked; k9/k10 are guarded by centered_row_coarse_cert.py.
    7: sp.Rational(3, 2),
    9: sp.Rational(3, 2),
    10: sp.Rational(3, 2),
}


def mixed_centered_or_certified_budget(k: int) -> tuple[sp.Expr, list[tuple[int, int, sp.Expr, sp.Expr, str]]]:
    """Use the sharper of the centered row-guarded bound and known low-k certificates."""
    h = sp.Pow(2, -sp.Rational(k, 2))
    k_center = sp.Rational(1, 60)
    derivative_bound = sp.Rational(1, 25)
    collar_bound = sp.Rational(9, 128)
    upper = sp.Integer(0)
    rows = []
    for j in range(0, (k - 1) // 2 + 1):
        reduced_k = k - 2 * j
        alpha = sp.Integer(1) if j == 0 else sp.Integer(2) ** j
        weight = sp.Integer(1) if j == 0 else sp.Pow(2, -sp.Rational(3 * (j - 1), 2))
        perturbation = h * h * (derivative_bound + alpha * alpha * (derivative_bound + 2 * collar_bound))
        centered = sp.sqrt(ROW_GUARD.get(reduced_k, sp.Rational(3, 2))) * sp.sqrt(k_center + perturbation)
        known = CERTIFIED_LOW_RESCALED.get(reduced_k)
        if known is not None and bool(sp.N(known - centered, 40) < 0):
            selected = known
            source = "known finite certificate"
        else:
            selected = centered
            source = f"centered row guard R<={sp.sstr(ROW_GUARD.get(reduced_k, sp.Rational(3, 2)))}"
        contribution = weight * selected
        upper += contribution
        rows.append((j, reduced_k, weight, contribution, source))
    return upper, rows


def print_mixed_budgets() -> None:
    print("\nSTEP 3  mixed centered/known-finite budgets for k8-k10")
    print("        low reduced blocks use existing certified C_res(k); centered rows use coarse row guards")
    for k in range(8, 11):
        upper, rows = mixed_centered_or_certified_budget(k)
        threshold = C2 * sp.Pow(2, sp.Rational(k, 2))
        margin = threshold - upper
        status = "fits" if bool(sp.N(margin, 40) > 0) else "misses"
        print(
            f"        k={k:2d}  U_mixed<={float(sp.N(upper, 30)):.12f}  "
            f"threshold={float(sp.N(threshold, 30)):.12f}  margin={float(sp.N(margin, 30)):+.12f}  {status}"
        )
        for j, reduced_k, weight, contribution, source in rows:
            print(
                f"          j={j}: reduced k={reduced_k:2d}  "
                f"weight={float(sp.N(weight, 20)):.12f}  "
                f"contrib={float(sp.N(contribution, 20)):.12f}  {source}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-bernstein", action="store_true")
    args = parser.parse_args()
    if not args.skip_bernstein:
        verify_centered_constants()
    print_budgets()
    print_mixed_budgets()


if __name__ == "__main__":
    main()
