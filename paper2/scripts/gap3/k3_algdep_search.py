#!/usr/bin/env python3
"""High-precision algebraic-dependence search for the k=3 reduced optimum."""

from __future__ import annotations

import os
import sys

import mpmath as mp
import sympy as sp

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.gap3.k3_stationary_system import (
    EQUIVARIANT_INITIAL,
    build_equivariant_expressions,
    build_expressions,
    stationarity_equations,
)


def solve_equivariant(prec: int = 220):
    symbols, B, X2, D2 = build_expressions()
    eq_symbols, B_eq, X2_eq, D2_eq = build_equivariant_expressions(symbols, B, X2, D2)
    equations = stationarity_equations(eq_symbols, B_eq, X2_eq, D2_eq)
    guess = [EQUIVARIANT_INITIAL[str(var)] for var in eq_symbols]
    sol = sp.nsolve(equations, eq_symbols, guess, tol=sp.Float(f"1e-{prec - 40}"), maxsteps=100, prec=prec)
    subs = dict(zip(eq_symbols, sol))
    value = sp.N(B_eq.subs(subs) / (X2_eq.subs(subs) * sp.sqrt(D2_eq.subs(subs))), prec - 20)
    return eq_symbols, sol, value


def pslq_powers(value: sp.Float, degree: int, digits: int, maxcoeff: int):
    mp.mp.dps = digits
    c = mp.mpf(str(value))
    return mp.pslq([c**i for i in range(degree + 1)], tol=mp.mpf(10) ** (-(digits - 30)), maxcoeff=maxcoeff)


def main() -> None:
    print("Solving 8-variable equivariant stationarity system at high precision...")
    symbols, sol, value = solve_equivariant()
    print("C3 =", sp.N(value, 90))
    print("variables:")
    for symbol, item in zip(symbols, sol):
        print(f"  {symbol} = {sp.N(item, 60)}")

    print("\nMinimal-polynomial PSLQ on C3:")
    for degree in (12, 16, 20, 24, 28, 32):
        rel = pslq_powers(value, degree=degree, digits=170, maxcoeff=10**18)
        print(f"  degree <= {degree:2d}, coeff <= 1e18: {rel}")

    c2 = sp.N(value * value, 190)
    print("\nMinimal-polynomial PSLQ on C3^2:")
    for degree in (12, 16, 20, 24, 28, 32):
        rel = pslq_powers(c2, degree=degree, digits=170, maxcoeff=10**18)
        print(f"  degree <= {degree:2d}, coeff <= 1e18: {rel}")


if __name__ == "__main__":
    main()