#!/usr/bin/env python3
"""Compress the reduced k=3 numerator into complex monomials."""

from __future__ import annotations

import os
import sys

import sympy as sp

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.gap3.k3_stationary_system import build_equivariant_expressions, build_expressions


def main() -> None:
    symbols, B, X2, D2 = build_expressions()
    eq_symbols, B_eq, X2_eq, D2_eq = build_equivariant_expressions(symbols, B, X2, D2)
    x1, y1, x3, y3, x5, y5, x7, y7 = eq_symbols
    numerator = sp.factor(-sp.Rational(455, 8) * B_eq)

    imaginary = sp.I
    u = x1 + imaginary * y1
    v = x3 + imaginary * y3
    w = x5 + imaginary * y5
    r = x7 + imaginary * y7

    monomial_sets = [
        ("uv", [u * v, u * sp.conjugate(v)]),
        (
            "uvw",
            [
                u * v * w,
                u * v * sp.conjugate(w),
                u * sp.conjugate(v) * w,
                sp.conjugate(u) * v * w,
            ],
        ),
        (
            "vwr",
            [
                v * w * r,
                v * w * sp.conjugate(r),
                v * sp.conjugate(w) * r,
                sp.conjugate(v) * w * r,
            ],
        ),
    ]

    unknowns = []
    trial = 0
    for label, monomials in monomial_sets:
        for index, monomial in enumerate(monomials):
            real_part, imag_part = sp.symbols(f"{label}{index}r {label}{index}i", real=True)
            unknowns.extend([real_part, imag_part])
            trial += sp.re((real_part + imaginary * imag_part) * monomial)

    trial = sp.expand_complex(trial)
    residual = sp.Poly(sp.expand(trial - numerator), x1, y1, x3, y3, x5, y5, x7, y7)
    equations = [sp.Eq(coefficient, 0) for coefficient in residual.coeffs()]
    solutions = sp.solve(equations, unknowns, dict=True, simplify=True)
    if not solutions:
        raise SystemExit("No complex compression found")

    solution = solutions[0]
    compressed = sp.expand_complex(trial.subs(solution))

    print("K3 reduced denominator:")
    print("A =", sp.factor(X2_eq / 8))
    print("G =", sp.factor(D2_eq / 16))
    print()
    print("Numerator bracket Phi = Re(...), coefficients:")
    for label, monomials in monomial_sets:
        print(label)
        for index, _ in enumerate(monomials):
            real_symbol = sp.symbols(f"{label}{index}r", real=True)
            imag_symbol = sp.symbols(f"{label}{index}i", real=True)
            coefficient = sp.simplify(solution.get(real_symbol, 0) + imaginary * solution.get(imag_symbol, 0))
            print(f"  c{index} = {coefficient}")
    print()
    print("Verification:", sp.simplify(compressed - numerator) == 0)
    print("Expanded numerator operation count:", sp.count_ops(numerator))
    print("Complex form operation count:", sp.count_ops(compressed))


if __name__ == "__main__":
    main()