#!/usr/bin/env python3
"""
route_a_joint_kernel_cert.py
============================
Reproducible audit for the Route A joint Schur-kernel sharpening.

This script checks four facts used in references/P11_CANCELLATION_BOUND.md and
references/SPECTRAL_TAXONOMY_TAIL_REPAIR.md:

1. The Hilbert-Schmidt mixed trace identity
       T_HS = q (s+c)(t+c) / rho.
2. The joint endpoint value
       K* = sqrt(2) (110 log(2) + 43) / 1280.
3. The Bernstein certificate for K(s,t) <= K* on [1,2]^2.
4. A k=12 finite-cell certificate using rational joint derivative/collar
    bounds on an enlarged square.
5. A k=11 weighted-row closure certificate using the certified all-row Schur
    inflation from k11_row_coarse_cert.py.
6. The rescaled closure constants and zero-error finite-height budgets.

The Bernstein step is exact after enclosing log(2) by
       693/1000 < log(2) < 347/500.
The resulting coefficients lie in Q(sqrt(2)); their signs are checked exactly.
"""

from __future__ import annotations

import argparse
import math
from math import comb
from typing import Dict, Tuple

import sympy as sp


LOG2_LO = sp.Rational(693, 1000)
LOG2_HI = sp.Rational(347, 500)
C2 = sp.Rational(22741865409341, 10**15)  # displayed C(I_2) used in tail_cert.py
K11_WEIGHTED_ROW_RATIO_UPPER = sp.Rational(1011760776695, 10**12)


def sign_qsqrt2(expr: sp.Expr) -> int:
    """Return the exact sign of an expression p + q*sqrt(2), p,q in Q."""
    root = sp.sqrt(2)
    expr = sp.expand(expr)
    pieces = sp.collect(expr, root, evaluate=False)
    p = sp.simplify(pieces.get(sp.Integer(1), sp.Integer(0)))
    q = sp.simplify(pieces.get(root, sp.Integer(0)))
    rest = sp.simplify(expr - p - q * root)
    if rest != 0:
        raise ValueError(f"expression is not in Q(sqrt(2)): {expr!r}; rest={rest!r}")
    if p.has(root) or q.has(root):
        raise ValueError(f"failed to decompose into Q(sqrt(2)): p={p!r}, q={q!r}")

    p = sp.Rational(p)
    q = sp.Rational(q)
    if q == 0:
        return 1 if p > 0 else (-1 if p < 0 else 0)
    if p == 0:
        return 1 if q > 0 else -1
    if p > 0 and q > 0:
        return 1
    if p < 0 and q < 0:
        return -1
    if q > 0:  # p < 0, compare q*sqrt(2) with -p
        cmp_expr = 2 * q * q - p * p
        return 1 if cmp_expr > 0 else (-1 if cmp_expr < 0 else 0)
    # q < 0 and p > 0, compare p with (-q)*sqrt(2)
    cmp_expr = p * p - 2 * q * q
    return 1 if cmp_expr > 0 else (-1 if cmp_expr < 0 else 0)


def verify_hs_trace_identity() -> None:
    """Symbolically verify T_HS = q(s+c)(t+c)/rho in adapted coordinates."""
    A, B, X, Y = sp.symbols("A B X Y", positive=True)
    xi = sp.Matrix([A, 0, 0])
    eta = sp.Matrix([B * X, B * Y, 0])
    zeta = -(xi + eta)
    rho = sp.expand(zeta.dot(zeta))
    projection = sp.eye(3) - zeta * zeta.T / rho

    basis_xi = [sp.Matrix([0, 1, 0]), sp.Matrix([0, 0, 1])]
    basis_eta = [sp.Matrix([-Y, X, 0]), sp.Matrix([0, 0, 1])]

    trace = sp.Integer(0)
    for avec in basis_xi:
        for bvec in basis_eta:
            a_oa = (eta.dot(avec)) * projection * bvec
            a_pmid = (zeta.dot(bvec)) * projection * avec
            trace += a_oa.dot(a_pmid)

    s = A**2
    t = B**2
    c = A * B * X
    q = 1 - X**2
    target = q * (s + c) * (t + c) / rho

    numerator = sp.together(trace - target).as_numer_denom()[0]
    numerator = sp.factor(numerator.subs(Y**2, 1 - X**2))
    if numerator != 0:
        raise AssertionError(f"Hilbert-Schmidt trace identity failed: {numerator}")

    print("STEP 1  Hilbert-Schmidt trace identity: PASS")
    print("        T_HS = q (s+c)(t+c) / rho")


def route_a_polynomial() -> Tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
    """Return (K, K_star, G, a, b, L) for the joint kernel endpoint certificate."""
    a, b, L = sp.symbols("a b L")
    s = a * a
    t = b * b
    root = sp.sqrt(2)

    p_oa = (
        5 * s**6 - 20 * s**5 * t + 30 * s**4 * t**2 + 150 * s**4
        - 20 * s**3 * t**3 - 80 * s**3 * t - 300 * s**3
        + 5 * s**2 * t**4 + 40 * s**2 * t**2 - 120 * s**2 * t + 350 * s**2
        + 40 * s * t**3 - 120 * s * t**2 + 280 * s * t
        - 20 * s * (s - t)**2 * (3 * s**2 + t**2) * L
        - 225 * s + 10 * t**4 - 60 * t**3 + 140 * t**2 - 150 * t + 62
    )
    m_oa = p_oa / (320 * a**7 * b**7)
    bracket = -1 + 2 * (s + t) * L + sp.Rational(1, 2) * (4 * s * t - (s + t)**2)
    m_pmid = (s - t)**2 / (8 * a**3 * b**5) * bracket
    k_joint = sp.expand(m_oa + m_pmid)
    k_star = root * (110 * L + 43) / 1280
    g_poly = sp.expand(1280 * a**7 * b**7 * (k_star - k_joint))
    return k_joint, k_star, g_poly, a, b, L


def power_to_bernstein_coefficients(poly: sp.Expr, x: sp.Symbol, y: sp.Symbol) -> Tuple[int, int, Dict[Tuple[int, int], sp.Expr]]:
    """Convert a bivariate power-basis polynomial to Bernstein coefficients."""
    power = sp.Poly(sp.expand(poly), x, y)
    deg_x = power.degree(x)
    deg_y = power.degree(y)
    coeffs = {(i, j): sp.expand(c) for (i, j), c in power.terms()}

    bernstein: Dict[Tuple[int, int], sp.Expr] = {}
    for k in range(deg_x + 1):
        for ell in range(deg_y + 1):
            value = sp.Integer(0)
            for i in range(k + 1):
                for j in range(ell + 1):
                    c = coeffs.get((i, j), sp.Integer(0))
                    if c == 0:
                        continue
                    value += (
                        c
                        * sp.Rational(comb(k, i), comb(deg_x, i))
                        * sp.Rational(comb(ell, j), comb(deg_y, j))
                    )
            bernstein[(k, ell)] = sp.expand(value)
    return deg_x, deg_y, bernstein


def lower_bound_linear_log2(expr: sp.Expr, log_symbol: sp.Symbol) -> sp.Expr:
    """Lower-bound A+B*log(2) using 693/1000 < log(2) < 347/500."""
    expr = sp.expand(expr)
    poly = sp.Poly(expr, log_symbol)
    if poly.degree() > 1:
        raise ValueError(f"expected an affine expression in log(2), got {expr!r}")
    slope = poly.coeff_monomial(log_symbol)
    base = poly.coeff_monomial(1)
    return sp.simplify(base + slope * (LOG2_LO if slope >= 0 else LOG2_HI))


def verify_nonnegative_bernstein_linear_log2(poly: sp.Expr, x: sp.Symbol, y: sp.Symbol, log_symbol: sp.Symbol, label: str) -> None:
    """Check all Bernstein lower coefficients are nonnegative for A+B*log(2)."""
    deg_x, deg_y, bernstein = power_to_bernstein_coefficients(poly, x, y)
    failures = []
    zeros = []
    min_positive = None
    for key, coeff in bernstein.items():
        lower = lower_bound_linear_log2(coeff, log_symbol)
        if lower < 0:
            failures.append((key, lower))
        elif lower == 0:
            zeros.append(key)
        else:
            numeric = float(sp.N(lower, 30))
            if min_positive is None or numeric < min_positive[1]:
                min_positive = (key, numeric)

    print(f"        {label}: bidegree=({deg_x},{deg_y}), zeros={zeros}")
    if min_positive is not None:
        print(f"          min positive lower coefficient {min_positive[1]:.12g} at {min_positive[0]}")
    if failures:
        for key, lower in failures[:10]:
            print(f"          FAILURE coefficient {key}: lower={lower}")
        raise AssertionError(f"{label}: {len(failures)} negative Bernstein lower coefficients")


def verify_bernstein_certificate(verbose: bool = False) -> None:
    """Verify K(s,t) <= K* on [1,2]^2 by Bernstein coefficients."""
    _, _, g_poly, a, b, L = route_a_polynomial()
    x, y = sp.symbols("x y")
    q = sp.sqrt(2) - 1
    square_poly = sp.expand(g_poly.subs({a: 1 + q * x, b: 1 + q * y}))
    deg_x, deg_y, bernstein = power_to_bernstein_coefficients(square_poly, x, y)

    zeros = []
    failures = []
    min_positive = None
    for key, coeff in bernstein.items():
        base = sp.expand(coeff.subs(L, 0))
        slope = sp.expand(sp.diff(coeff, L))
        slope_sign = sign_qsqrt2(slope)
        lower = sp.expand(base + slope * (LOG2_LO if slope_sign >= 0 else LOG2_HI))
        lower_sign = sign_qsqrt2(lower)
        if lower_sign < 0:
            failures.append((key, lower))
        elif lower_sign == 0:
            zeros.append(key)
        else:
            numeric = float(sp.N(lower, 40))
            if min_positive is None or numeric < min_positive[1]:
                min_positive = (key, numeric, lower)

    print("STEP 2  Joint-kernel Bernstein endpoint certificate")
    print(f"        bidegree: ({deg_x},{deg_y})")
    print(f"        coefficients checked: {(deg_x + 1) * (deg_y + 1)}")
    print(f"        zero lower coefficients: {zeros}")
    if min_positive is not None:
        print(f"        minimum positive lower coefficient: {min_positive[1]:.12g} at {min_positive[0]}")
    if failures:
        for key, lower in failures[:10]:
            print(f"        FAILURE coefficient {key}: lower={lower}")
        raise AssertionError(f"{len(failures)} Bernstein lower coefficients are negative")
    if zeros != [(deg_x, 0)]:
        raise AssertionError(f"unexpected zero coefficient set: {zeros}")
    if verbose:
        print("        endpoint zero corresponds to x=1,y=0, i.e. s=2,t=1")
    print("        PASS: K(s,t) <= K* on [1,2]^2")


def verify_k12_finite_cell_certificate() -> None:
    """Certify a finite k=12 joint-kernel envelope with rational constants."""
    k_joint, k_star, _, a, b, L = route_a_polynomial()
    x, y = sp.symbols("x y")
    s = a * a
    t = b * b

    # The k=12 thickening has s,t in [1-eps,2+eps], eps=(sqrt(6)+3h/4)h,
    # h=1/64.  This is contained in a^2,b^2 with a,b in [49/50,10/7].
    h12 = sp.Rational(1, 64)
    c12 = sp.sqrt(6) + sp.Rational(3, 4) * h12
    eps12 = c12 * h12
    a_lo = sp.Rational(49, 50)
    a_hi = sp.Rational(10, 7)
    if not bool(sp.N(a_lo**2 - (1 - eps12), 40) < 0):
        raise AssertionError("lower rational square does not cover the k=12 thickening")
    if not bool(sp.N(a_hi**2 - (2 + eps12), 40) > 0):
        raise AssertionError("upper rational square does not cover the k=12 thickening")
    transform = {a: a_lo + (a_hi - a_lo) * x, b: a_lo + (a_hi - a_lo) * y}

    ds = sp.diff(k_joint, a) / (2 * a)
    dt = sp.diff(k_joint, b) / (2 * b)
    derivative_checks = [
        ("0.4 - partial_s K", sp.Rational(2, 5) - ds),
        ("0.4 + partial_s K", sp.Rational(2, 5) + ds),
        ("0.6 - partial_t K", sp.Rational(3, 5) - dt),
        ("0.6 + partial_t K", sp.Rational(3, 5) + dt),
    ]

    def g_joint(rho_value: int) -> sp.Expr:
        rho = sp.Rational(rho_value)
        disc = 4 * s * t - (rho - s - t) ** 2
        g_oa = (s - rho) ** 2 * disc**2 / (32 * a**7 * b**7 * rho**2)
        g_pmid = (s - t) ** 2 * disc / (8 * a**3 * b**5 * rho**2)
        return sp.expand(g_oa + g_pmid)

    collar_checks = [
        ("0.3 - collar K at rho=1", sp.Rational(3, 10) - g_joint(1)),
        ("0.3 - collar K at rho=2", sp.Rational(3, 10) - g_joint(2)),
    ]

    print("STEP 3  k=12 finite-cell joint certificate")
    print("        enlarged square: a,b in [49/50,10/7]")
    print("        proves |partial_s K| <= 0.4, |partial_t K| <= 0.6, collar <= 0.3")
    for label, expr in derivative_checks + collar_checks:
        numerator = sp.together(expr).as_numer_denom()[0]
        poly = sp.expand(numerator.subs(transform))
        verify_nonnegative_bernstein_linear_log2(poly, x, y, L, label)

    ds_bound = sp.Rational(2, 5)
    dt_bound = sp.Rational(3, 5)
    collar_bound = sp.Rational(3, 10)
    k_star_hi = sp.sqrt(2) * (110 * LOG2_HI + 43) / 1280

    def b_joint(alpha: sp.Expr) -> sp.Expr:
        return c12 * (ds_bound + alpha * (dt_bound + 2 * collar_bound))

    def p_joint(alpha: sp.Expr) -> sp.Expr:
        return sp.sqrt(k_star_hi + b_joint(alpha) * h12)

    j_max = 24
    finite_sum = p_joint(1)
    for j in range(1, j_max + 1):
        finite_sum += sp.Pow(2, -sp.Rational(3 * (j - 1), 2)) * p_joint(2**j)

    b0 = c12 * ds_bound
    b1 = c12 * (dt_bound + 2 * collar_bound)
    tail_constant = sp.sqrt(k_star_hi + b0 * h12) * sp.Pow(2, -sp.Rational(3 * j_max, 2)) / (1 - sp.Pow(2, -sp.Rational(3, 2)))
    tail_variable = sp.sqrt(b1 * h12) * sp.Pow(2, sp.Rational(3, 2)) * sp.Pow(2, -j_max)
    upper = finite_sum + tail_constant + tail_variable
    threshold = C2 * 64
    margin = threshold - upper

    print("        finite envelope uses K*_upper with log(2) <= 347/500")
    print(f"        U_12 <= {float(sp.N(upper, 30)):.12f}")
    print(f"        C(I_2)*2^6 = {float(sp.N(threshold, 30)):.12f}")
    print(f"        margin >= {float(sp.N(margin, 30)):.12f}")
    if not bool(sp.N(margin, 30) > 0):
        raise AssertionError("k=12 finite envelope does not close")
    print("        PASS: k=12 finite-cell Route A envelope closes")


def diagnose_k11_cell_obstruction() -> None:
    """Certify and print the k=11 miss for the same finite-cell strategy."""
    k_joint, _, _, a, b, L = route_a_polynomial()
    x, y = sp.symbols("x y")
    s = a * a
    t = b * b

    # The k=11 thickening is covered by a,b in [19/20,3/2].  On this square,
    # simple Bernstein-certified bounds are enough to show the current cell
    # strategy is far too lossy for k=11.
    k = 11
    h = sp.Pow(2, -sp.Rational(k, 2))
    c = sp.sqrt(6) + sp.Rational(3, 4) * h
    eps = c * h
    a_lo = sp.Rational(19, 20)
    a_hi = sp.Rational(3, 2)
    if not bool(sp.N(a_lo**2 - (1 - eps), 40) < 0):
        raise AssertionError("lower k=11 diagnostic square does not cover the thickening")
    if not bool(sp.N(a_hi**2 - (2 + eps), 40) > 0):
        raise AssertionError("upper k=11 diagnostic square does not cover the thickening")
    transform = {a: a_lo + (a_hi - a_lo) * x, b: a_lo + (a_hi - a_lo) * y}

    ds = sp.diff(k_joint, a) / (2 * a)
    dt = sp.diff(k_joint, b) / (2 * b)

    def g_joint(rho_value: int) -> sp.Expr:
        rho = sp.Rational(rho_value)
        disc = 4 * s * t - (rho - s - t) ** 2
        g_oa = (s - rho) ** 2 * disc**2 / (32 * a**7 * b**7 * rho**2)
        g_pmid = (s - t) ** 2 * disc / (8 * a**3 * b**5 * rho**2)
        return sp.expand(g_oa + g_pmid)

    checks = [
        ("0.5 - partial_s K", sp.Rational(1, 2) - ds),
        ("0.5 + partial_s K", sp.Rational(1, 2) + ds),
        ("0.75 - partial_t K", sp.Rational(3, 4) - dt),
        ("0.75 + partial_t K", sp.Rational(3, 4) + dt),
        ("0.4 - collar K at rho=1", sp.Rational(2, 5) - g_joint(1)),
        ("0.4 - collar K at rho=2", sp.Rational(2, 5) - g_joint(2)),
    ]

    print("STEP X  k=11 finite-cell obstruction diagnostic")
    print("        enlarged square: a,b in [19/20,3/2]")
    print("        proves |partial_s K| <= 0.5, |partial_t K| <= 0.75, collar <= 0.4")
    for label, expr in checks:
        numerator = sp.together(expr).as_numer_denom()[0]
        poly = sp.expand(numerator.subs(transform))
        verify_nonnegative_bernstein_linear_log2(poly, x, y, L, label)

    k_star_hi = sp.sqrt(2) * (110 * LOG2_HI + 43) / 1280

    def b_joint(alpha: sp.Expr) -> sp.Expr:
        return c * (sp.Rational(1, 2) + alpha * (sp.Rational(3, 4) + 2 * sp.Rational(2, 5)))

    def p_joint(alpha: sp.Expr) -> sp.Expr:
        return sp.sqrt(k_star_hi + b_joint(alpha) * h)

    upper = p_joint(1)
    for j in range(1, (k - 1) // 2 + 1):
        upper += sp.Pow(2, -sp.Rational(3 * (j - 1), 2)) * p_joint(2**j)

    threshold = C2 * sp.Pow(2, sp.Rational(k, 2))
    margin = threshold - upper

    print(f"        U_11 <= {float(sp.N(upper, 30)):.12f}")
    print(f"        C(I_2)*2^(11/2) = {float(sp.N(threshold, 30)):.12f}")
    print(f"        margin = {float(sp.N(margin, 30)):.12f}")
    print("        CONCLUSION: this finite-cell strategy cannot close k=11 without a sharper idea")


def k11_radial_envelope() -> tuple[sp.Expr, sp.Expr, sp.Expr, list[tuple[int, sp.Expr, sp.Expr, sp.Expr]]]:
    """Return the k=11 radial shell/rho envelope with certified coarse constants."""
    k = 11
    h = sp.Pow(2, -sp.Rational(k, 2))
    k_star_hi = sp.sqrt(2) * (110 * LOG2_HI + 43) / 1280
    sqrt_k = sp.sqrt(k_star_hi)
    ds_bound = sp.Rational(1, 2)
    dt_bound = sp.Rational(3, 4)
    collar_bound = sp.Rational(2, 5)

    upper = sp.Integer(0)
    zero = sp.Integer(0)
    rows = []
    for j in range(0, (k - 1) // 2 + 1):
        alpha = sp.Integer(1) if j == 0 else sp.Integer(2) ** j
        weight = sp.Integer(1) if j == 0 else sp.Pow(2, -sp.Rational(3 * (j - 1), 2))
        perturbation = h * h * (ds_bound + alpha * alpha * (dt_bound + 2 * collar_bound))
        p_joint = sp.sqrt(k_star_hi + perturbation)
        zero_contrib = weight * sqrt_k
        upper_contrib = weight * p_joint
        zero += zero_contrib
        upper += upper_contrib
        rows.append((j, zero_contrib, upper_contrib, upper_contrib - zero_contrib))

    threshold = C2 * sp.Pow(2, sp.Rational(k, 2))
    return threshold, zero, upper, rows


def diagnose_k11_radial_target() -> None:
    """Print the conditional k=11 target obtained from radial shell spacing.

    This does not prove the shell/rho finite-sum lemma.  It records that the
    already-certified k=11 derivative/collar constants are numerically strong
    enough once the discretisation error is charged at squared-radius shell
    spacing h^2 rather than vector-cube spacing h.
    """
    threshold, zero, upper, rows = k11_radial_envelope()
    margin = threshold - upper

    print("STEP Y  k=11 radial shell/rho target diagnostic")
    print("        assumes a shell/rho finite-sum lemma with squared-radius mesh")
    print("        uses certified coarse bounds |partial_s K|<=1/2, |partial_t K|<=3/4, collar<=2/5")
    print(f"        zero-error finite-height = {float(sp.N(zero, 30)):.12f}")
    print(f"        U_11,radial <= {float(sp.N(upper, 30)):.12f}")
    print(f"        C(I_2)*2^(11/2) = {float(sp.N(threshold, 30)):.12f}")
    print(f"        conditional margin = {float(sp.N(margin, 30)):.12f}")
    for j, zero_contrib, upper_contrib, loss in rows:
        print(
            f"          j={j}: zero={float(sp.N(zero_contrib, 20)):.12f}  "
            f"radial={float(sp.N(upper_contrib, 20)):.12f}  "
            f"loss={float(sp.N(loss, 20)):.12f}"
        )
    if not bool(sp.N(margin, 30) > 0):
        raise AssertionError("conditional radial k=11 target does not close")
    print("        PASS CONDITIONAL: radial shell/rho discretisation would close k=11")


def certify_k11_weighted_row_closure() -> None:
    """Certify k=11 Route A closure from radial envelope plus row domination."""
    threshold, zero, radial_upper, rows = k11_radial_envelope()
    row_ratio = K11_WEIGHTED_ROW_RATIO_UPPER
    row_factor = sp.sqrt(row_ratio)
    weighted_upper = row_factor * radial_upper
    row_ratio_budget = sp.simplify((threshold / radial_upper) ** 2)
    ratio_margin = row_ratio_budget - row_ratio
    margin = threshold - weighted_upper

    print("STEP Z  k=11 weighted-row Route A closure certificate")
    print("        radial shell/rho envelope uses h^2 and (2^j h)^2 mesh")
    print("        row Schur domination uses all-row coarse guard from k11_row_coarse_cert.py")
    print(f"        zero-error finite-height = {float(sp.N(zero, 30)):.12f}")
    print(f"        U_11,radial <= {float(sp.N(radial_upper, 30)):.12f}")
    print(f"        row ratio upper R_11 <= {float(sp.N(row_ratio, 30)):.12f}")
    print(f"        allowed row ratio <= {float(sp.N(row_ratio_budget, 30)):.12f}")
    print(f"        row-ratio margin >= {float(sp.N(ratio_margin, 30)):.12f}")
    print(f"        sqrt(R_11) * U_11,radial <= {float(sp.N(weighted_upper, 30)):.12f}")
    print(f"        C(I_2)*2^(11/2) = {float(sp.N(threshold, 30)):.12f}")
    print(f"        final k=11 margin >= {float(sp.N(margin, 30)):.12f}")
    for j, zero_contrib, radial_contrib, loss in rows:
        weighted_contrib = row_factor * radial_contrib
        print(
            f"          j={j}: radial={float(sp.N(radial_contrib, 20)):.12f}  "
            f"weighted={float(sp.N(weighted_contrib, 20)):.12f}  "
            f"radial_loss={float(sp.N(loss, 20)):.12f}"
        )
    if not bool(sp.N(ratio_margin, 30) > 0):
        raise AssertionError("k=11 row ratio exceeds radial squared-ratio budget")
    if not bool(sp.N(margin, 30) > 0):
        raise AssertionError("k=11 weighted-row Route A envelope does not close")
    print("        PASS: k=11 weighted-row Route A envelope closes")


def print_constants() -> None:
    """Print endpoint and closure constants."""
    _, k_star, _, _, _, L = route_a_polynomial()
    k_star_log = sp.simplify(k_star.subs(L, sp.log(2)))
    c_joint = sp.sqrt(k_star_log)
    multiplier = 1 / (1 - 2 ** (-sp.Rational(3, 2)))
    old_primitive = sp.sqrt(sp.Rational(17, 320)) + sp.sqrt(3 * sp.sqrt(2) * (sp.log(16) - 1) / 64)
    relay_only = old_primitive + c_joint * multiplier
    primitive_plus_relay = c_joint * (1 + multiplier)

    print("STEP 4  Route A constants")
    print(f"        K* = {sp.sstr(k_star)}")
    print(f"        sqrt(K*) = {float(sp.N(c_joint, 30)):.15f}")
    print(f"        relay multiplier = {float(sp.N(multiplier, 30)):.15f}")
    print(f"        old primitive + joint relay = {float(sp.N(relay_only, 30)):.15f}")
    print(f"        joint primitive + joint relay = {float(sp.N(primitive_plus_relay, 30)):.15f}")

    print("\nSTEP 5  Zero-error finite-height budgets")
    for k in range(2, 13):
        threshold = float(sp.N(C2 * 2 ** (sp.Rational(k, 2)), 30))
        jmax = max(0, (k - 1) // 2)
        weights = 1.0 + sum(2 ** (-1.5 * (j - 1)) for j in range(1, jmax + 1))
        asym = float(sp.N(c_joint, 30)) * weights
        budget = threshold - asym
        status = "fits" if budget > 0 else "misses"
        print(
            f"        k={k:2d}  threshold={threshold:.12f}  "
            f"joint finite-height={asym:.12f}  budget={budget:+.12f}  {status}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="print extra certificate details")
    parser.add_argument("--skip-bernstein", action="store_true", help="skip the symbolic Bernstein certificate")
    parser.add_argument("--skip-k12-finite", action="store_true", help="skip the symbolic k=12 finite-cell certificate")
    parser.add_argument("--diagnose-k11", action="store_true", help="also run the k=11 finite-cell obstruction diagnostic")
    parser.add_argument("--diagnose-k11-radial", action="store_true", help="also print the conditional k=11 radial shell/rho target")
    parser.add_argument("--certify-k11-weighted-row", action="store_true", help="certify k=11 using the weighted row-sum domination guard")
    args = parser.parse_args()

    print("=" * 72)
    print("Route A joint Schur-kernel certificate")
    print("=" * 72)
    verify_hs_trace_identity()
    print()
    if not args.skip_bernstein:
        verify_bernstein_certificate(verbose=args.verbose)
        print()
    if not args.skip_k12_finite:
        verify_k12_finite_cell_certificate()
        print()
    if args.diagnose_k11:
        diagnose_k11_cell_obstruction()
        print()
    if args.diagnose_k11_radial:
        diagnose_k11_radial_target()
        print()
    if args.certify_k11_weighted_row:
        certify_k11_weighted_row_closure()
        print()
    print_constants()
    print()
    print("ALL REQUESTED CHECKS PASSED")


if __name__ == "__main__":
    main()
