#!/usr/bin/env python3
"""Algebraic/radical probes for the reduced k=3 optimum.

This script is research diagnostic code.  It does not certify the k=3
maximum.  It asks two narrower questions:

1. Does the high-precision k=3 value satisfy a small algebraic relation over
   the natural coefficient field Q(sqrt(2), sqrt(5), sqrt(7))?
2. What is the exact polynomial stationarity system one would feed into a
   real-root isolation / elimination proof?

The second output is intentionally structural: full Groebner/resultant
elimination is likely too large for a default run, but the equations are the
right starting point for that proof.
"""

from __future__ import annotations

import argparse
import os
import sys

import mpmath as mp
import sympy as sp

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.gap3.k3_algdep_search import solve_equivariant


def natural_field_basis_mp(digits: int, include_sqrt13: bool = False) -> list[mp.mpf]:
    mp.mp.dps = digits
    radicands = [2, 5, 7] + ([13] if include_sqrt13 else [])
    basis = []
    for mask in range(1 << len(radicands)):
        value = mp.mpf(1)
        for index, radicand in enumerate(radicands):
            if mask & (1 << index):
                value *= mp.sqrt(radicand)
        basis.append(value)
    return basis


def natural_field_basis_names(include_sqrt13: bool = False) -> list[str]:
    radicands = [2, 5, 7] + ([13] if include_sqrt13 else [])
    names = []
    for mask in range(1 << len(radicands)):
        factors = [str(radicand) for index, radicand in enumerate(radicands) if mask & (1 << index)]
        if not factors:
            names.append("1")
        else:
            product = 1
            for factor in factors:
                product *= int(factor)
            names.append(f"sqrt{product}")
    return names


def pslq_over_natural_field(value: mp.mpf, degree: int, digits: int, maxcoeff: int, include_sqrt13: bool = False):
    """Search sum_{i,j} a_{ij} basis_j value^i = 0."""
    mp.mp.dps = digits
    basis = natural_field_basis_mp(digits, include_sqrt13=include_sqrt13)
    vector = []
    for power in range(degree + 1):
        value_power = value**power
        vector.extend(field_element * value_power for field_element in basis)
    tolerance = mp.mpf(10) ** (-(digits - 30))
    return mp.pslq(vector, tol=tolerance, maxcoeff=maxcoeff, maxsteps=1000)


def format_field_relation(relation: list[int], symbol: str = "x", include_sqrt13: bool = False) -> str:
    basis_names = natural_field_basis_names(include_sqrt13=include_sqrt13)
    terms = []
    for index, coefficient in enumerate(relation):
        if coefficient == 0:
            continue
        power, basis_index = divmod(index, len(basis_names))
        factor = basis_names[basis_index]
        monomial = "1" if power == 0 else symbol if power == 1 else f"{symbol}^{power}"
        if factor != "1":
            monomial = f"{factor}*{monomial}"
        terms.append(f"{coefficient}*{monomial}")
    return " + ".join(terms) + " = 0" if terms else "0 = 0"


def run_field_pslq(value: sp.Float, digits: int, max_degree: int, maxcoeff: int, include_sqrt13: bool = False) -> None:
    mp.mp.dps = digits
    c_value = mp.mpf(str(sp.N(value, digits - 20)))
    c2_value = c_value * c_value
    scaled_c2 = 1120 * c2_value

    targets = [
        ("C3", c_value),
        ("C3^2", c2_value),
        ("1120*C3^2", scaled_c2),
    ]
    for label, target in targets:
        field_label = "Q(sqrt2,sqrt5,sqrt7,sqrt13)" if include_sqrt13 else "Q(sqrt2,sqrt5,sqrt7)"
        print(f"\nPSLQ over {field_label} for {label}:")
        for degree in range(1, max_degree + 1):
            relation = pslq_over_natural_field(target, degree, digits, maxcoeff, include_sqrt13=include_sqrt13)
            print(f"  degree <= {degree:2d}, coeff <= {maxcoeff:g}: {relation}")
            if relation:
                print("  relation:", format_field_relation(relation, "x", include_sqrt13=include_sqrt13))
                break


def p_sigma_expr(sigma: int, radius: sp.Symbol, cos_theta: sp.Symbol, sin_theta: sp.Symbol) -> sp.Expr:
    return (
        sp.Integer(1259)
        - 108 * sigma * sp.sqrt(35)
        + 637 * radius**2
        + 2
        * radius
        * (
            (162 * sigma * sp.sqrt(7) - 299 * sp.sqrt(5)) * cos_theta
            - (180 * sp.sqrt(2) + 39 * sigma * sp.sqrt(70)) * sin_theta
        )
    )


def q_theta_expr(cos_theta: sp.Symbol, sin_theta: sp.Symbol) -> sp.Expr:
    return 575 + 325 * (cos_theta**2 - sin_theta**2) - 300 * sp.sqrt(10) * cos_theta * sin_theta


def build_stationary_polynomial_system() -> tuple[list[sp.Symbol], list[sp.Expr]]:
    """Build exact polynomial equations for the five-variable kernel.

    Variables are p,q,r,h,c,s,a,b,e,m,w, where c=cos(theta), s=sin(theta),
    a^2=P_+, b^2=P_-, e^2=Q, m is the numerator bracket, and w=C3^2.
    """
    p, q, r, h, c, s, a, b, e, m, w = sp.symbols("p q r h c s a b e m w", real=True)
    sqrt2 = sp.sqrt(2)

    area = 2 + 5 * p**2 + 7 * q**2 + 5 * r**2 + 7 * h**2
    diss = 8 + 25 * p**2 + 49 * q**2 + 25 * r**2 + 49 * h**2
    p_plus = p_sigma_expr(1, r, c, s)
    p_minus = p_sigma_expr(-1, r, c, s)
    qth = q_theta_expr(c, s)

    bracket = p * (a + b) + sqrt2 * h * r * e

    equations: list[sp.Expr] = [
        c**2 + s**2 - 1,
        a**2 - p_plus,
        b**2 - p_minus,
        e**2 - qth,
        m - bracket,
        280**2 * w * area**2 * diss - 70 * q**2 * m**2,
    ]

    # log-stationarity in p, q, h.  These are polynomial after multiplying by
    # the common denominator area*diss*m.
    equations.extend(
        [
            (a + b) * area * diss - m * p * (10 * diss + 25 * area),
            area * diss - q**2 * (14 * diss + 49 * area),
            sqrt2 * r * e * area * diss - m * h * (14 * diss + 49 * area),
        ]
    )

    # r-stationarity.  Use a_r=P_r/(2a), b_r=P_r/(2b), e_r=0 and clear 2ab.
    p_plus_r = sp.diff(p_plus, r)
    p_minus_r = sp.diff(p_minus, r)
    m_r_cleared = p * (p_plus_r * b + p_minus_r * a) + 2 * sqrt2 * h * e * a * b
    equations.append(
        m_r_cleared * area * diss - 2 * a * b * m * r * (10 * diss + 25 * area)
    )

    # theta-stationarity: -s d/dc + c d/ds of log(K).  area and diss are
    # theta-independent, so only m contributes.  Clear 2abe.
    p_plus_theta = -s * sp.diff(p_plus, c) + c * sp.diff(p_plus, s)
    p_minus_theta = -s * sp.diff(p_minus, c) + c * sp.diff(p_minus, s)
    qth_theta = -s * sp.diff(qth, c) + c * sp.diff(qth, s)
    equations.append(
        p * e * (p_plus_theta * b + p_minus_theta * a) + sqrt2 * h * r * a * b * qth_theta
    )

    variables = [p, q, r, h, c, s, a, b, e, m, w]
    return variables, [sp.factor(sp.expand(equation)) for equation in equations]


def run_stationary_system_summary(value: sp.Float) -> None:
    variables, equations = build_stationary_polynomial_system()
    print("\nExact stationary polynomial system for the five-variable k3 kernel:")
    print("  variables:", ", ".join(str(var) for var in variables))
    print(f"  equations: {len(equations)}")
    for index, equation in enumerate(equations, 1):
        poly = sp.Poly(equation, *variables, extension=[sp.sqrt(2), sp.sqrt(5), sp.sqrt(7)])
        print(
            f"  E{index:02d}: total_degree={poly.total_degree():2d}, "
            f"terms={len(poly.terms()):4d}, ops={sp.count_ops(equation):5d}"
        )

    print("\n  value relation uses w = C3^2:")
    print("    280^2*w*A^2*G = 70*q^2*m^2")
    print(f"    numerical w ~= {sp.N(value * value, 60)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digits", type=int, default=180)
    parser.add_argument("--max-degree", type=int, default=8)
    parser.add_argument("--maxcoeff", type=int, default=10**14)
    parser.add_argument("--include-sqrt13", action="store_true", help="search over Q(sqrt2,sqrt5,sqrt7,sqrt13)")
    parser.add_argument("--skip-nsolve", action="store_true", help="Use the documented k3 value instead of solving.")
    args = parser.parse_args()

    if args.skip_nsolve:
        value = sp.Float("0.021936469459403747249299192478957700397867315103825", args.digits)
    else:
        print(f"Solving equivariant k3 stationarity system at {args.digits} digits...")
        _, _, value = solve_equivariant(prec=max(args.digits + 30, 120))

    print("\nC3 value used:")
    print(" ", sp.N(value, min(args.digits - 20, 100)))
    print("C3^2:")
    print(" ", sp.N(value * value, min(args.digits - 20, 100)))

    run_field_pslq(value, args.digits, args.max_degree, args.maxcoeff, include_sqrt13=args.include_sqrt13)
    run_stationary_system_summary(value)


if __name__ == "__main__":
    main()