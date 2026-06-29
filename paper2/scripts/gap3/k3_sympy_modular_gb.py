#!/usr/bin/env python3
"""Local modular Groebner probe for the equivariant k=3 closed-form system.

This is the fallback when Sage/Singular is not available.  It specializes the
quadratic radicals in the 8-variable equivariant stationary equations modulo a
prime, adds W X^4 D^2 - B^2 = 0 for W=C3^2, and asks SymPy for a modular
Groebner basis.
"""

from __future__ import annotations

import argparse
import os
import sys
from math import prod

import sympy as sp

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.gap3.k3_stationary_system import (  # noqa: E402
    build_equivariant_expressions,
    build_expressions,
    stationarity_equations,
)


RADICANDS = (2, 5, 7, 13)
COMPOSITE_RADICANDS = (10, 14, 26, 35, 65, 70, 91, 130, 182, 455, 910)


def is_residue(prime: int, n: int) -> bool:
    return pow(n % prime, (prime - 1) // 2, prime) == 1


def find_prime(start: int = 1009, stop: int = 20000) -> int:
    for candidate in range(start, stop):
        if candidate < 2:
            continue
        if not sp.isprime(candidate):
            continue
        if all(is_residue(candidate, n) for n in RADICANDS):
            return candidate
    raise RuntimeError("no suitable prime found")


def sqrt_mod_first(n: int, prime: int) -> int:
    roots = sp.sqrt_mod(n % prime, prime, all_roots=True)
    if not roots:
        raise ValueError(f"{n} is not a quadratic residue modulo {prime}")
    return int(roots[0])


def radical_substitution(prime: int) -> dict:
    elementary = {n: sqrt_mod_first(n, prime) for n in RADICANDS}
    substitutions = {sp.sqrt(n): value for n, value in elementary.items()}
    factors = {
        10: (2, 5),
        14: (2, 7),
        26: (2, 13),
        35: (5, 7),
        65: (5, 13),
        70: (2, 5, 7),
        91: (7, 13),
        130: (2, 5, 13),
        182: (2, 7, 13),
        455: (5, 7, 13),
        910: (2, 5, 7, 13),
    }
    for radicand in COMPOSITE_RADICANDS:
        substitutions[sp.sqrt(radicand)] = prod(elementary[item] for item in factors[radicand]) % prime
    return substitutions


def build_modular_polys(prime: int):
    symbols, b_expr, x2_expr, d2_expr = build_expressions()
    eq_symbols, b_eq, x2_eq, d2_eq = build_equivariant_expressions(symbols, b_expr, x2_expr, d2_expr)
    w = sp.Symbol("W")
    equations = stationarity_equations(eq_symbols, b_eq, x2_eq, d2_eq)
    equations.append(w * x2_eq**2 * d2_eq - b_eq**2)
    substitutions = radical_substitution(prime)
    gens = tuple(eq_symbols) + (w,)
    polynomials = []
    for equation in equations:
        numerator = sp.together(equation.xreplace(substitutions)).as_numer_denom()[0]
        polynomials.append(sp.Poly(sp.expand(numerator), *gens, modulus=prime))
    return gens, polynomials, substitutions


def summarize(gens, polynomials) -> None:
    print("variables:", ", ".join(str(g) for g in gens))
    print(f"equations: {len(polynomials)}")
    for index, polynomial in enumerate(polynomials, 1):
        print(
            f"  E{index:02d}: degree={polynomial.total_degree():2d} "
            f"terms={len(polynomial.terms()):5d}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prime", type=int, default=0)
    parser.add_argument("--order", choices=("grevlex", "lex"), default="grevlex")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--max-print", type=int, default=20)
    args = parser.parse_args()

    prime = args.prime or find_prime()
    print(f"Building SymPy modular system over GF({prime})")
    gens, polynomials, substitutions = build_modular_polys(prime)
    print(
        "sqrt choices:",
        ", ".join(f"sqrt({key.args[0]})={value}" for key, value in sorted(substitutions.items(), key=lambda item: int(item[0].args[0]))),
    )
    summarize(gens, polynomials)
    if args.summary_only:
        return

    print(f"\nComputing {args.order} Groebner basis ...")
    basis = sp.groebner([polynomial.as_expr() for polynomial in polynomials], *gens, modulus=prime, order=args.order)
    print(f"basis length: {len(basis.polys)}")
    w = gens[-1]
    w_only = []
    for polynomial in basis.polys:
        expr = polynomial.as_expr()
        free = expr.free_symbols
        if free and free.issubset({w}):
            w_only.append(sp.Poly(expr, w, modulus=prime))
    print(f"W-only generators: {len(w_only)}")
    for index, polynomial in enumerate(w_only, 1):
        print(f"  W{index}: degree={polynomial.degree()}, terms={len(polynomial.terms())}")
        print(f"    {polynomial.as_expr()}")
    if not w_only:
        for index, polynomial in enumerate(basis.polys[: args.max_print], 1):
            print(
                f"  G{index:02d}: degree={polynomial.total_degree():2d} "
                f"terms={len(polynomial.terms()):5d} lm={polynomial.LM(order=args.order)}"
            )


if __name__ == "__main__":
    main()