#!/usr/bin/env python3
"""Exact stationarity system for the k=3 9-mode DCxA support.

This fixes the observed phase/sign pattern and writes the optimiser as a real
algebraic problem.  It does not recertify anything; it is a closed-form probe:
the constant is the value of an exact cubic/quadratic variational system.
"""

from __future__ import annotations

import math
import os
import sys

import sympy as sp
import mpmath as mp

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


MODES = [
    (2, 2, 0),
    (1, 0, -3),
    (0, 1, -3),
    (1, 2, 3),
    (2, 1, 3),
    (3, 1, 0),
    (1, 3, 0),
    (2, -1, -3),
    (1, -2, 3),
]


# Initial values from scripts/gap3/k3_closed_form_probe.py.
INITIAL = {
    "x1": 0.43403360506775296**0.5 * math.cos(0.964295497102960),
    "y1": 0.43403360506775296**0.5 * math.sin(0.964295497102960),
    "x2": 0.43403374729289310**0.5 * math.cos(0.606501001304135),
    "y2": 0.43403374729289310**0.5 * math.sin(0.606501001304135),
    "x3": 0.27608011524945320**0.5 * math.cos(0.726127081308934),
    "y3": 0.27608011524945320**0.5 * math.sin(0.726127081308934),
    "x4": 0.27608016372983710**0.5 * math.cos(0.668345359470302),
    "y4": 0.27608016372983710**0.5 * math.sin(0.668345359470302),
    "x5": 0.06177023238699261**0.5 * math.cos(0.472106605995880),
    "y5": 0.06177023238699261**0.5 * math.sin(0.472106605995880),
    "x6": 0.06177030271210525**0.5 * math.cos(0.472106188377501),
    "y6": 0.06177030271210525**0.5 * math.sin(0.472106188377501),
    "x7": 0.00131274408800750**0.5 * math.cos(0.212746580322321),
    "y7": 0.00131274408800750**0.5 * math.sin(0.212746580322321),
    "x8": 0.00131275812237912**0.5 * math.cos(1.534373016904390),
    "y8": 0.00131275812237912**0.5 * math.sin(1.534373016904390),
}

REDUCED_INITIAL = {
    "x1": 0.37552035165988239970,
    "y1": 0.54131151348925115975,
    "x3": 0.39289355501814401177,
    "y3": 0.34887649048013801224,
    "x4": 0.41238531365976324740,
    "y4": 0.32560482837677924804,
    "x5": -0.22134949774847097783,
    "y5": -0.11302519887743256697,
    "x7": 0.03541513479120968952,
    "y7": 0.00765022308804324371,
    "x8": 0.00131938219943736099,
    "y8": 0.03620796757881045047,
}

EQUIVARIANT_INITIAL = {
    "x1": REDUCED_INITIAL["x1"],
    "y1": REDUCED_INITIAL["y1"],
    "x3": REDUCED_INITIAL["x3"],
    "y3": REDUCED_INITIAL["y3"],
    "x5": REDUCED_INITIAL["x5"],
    "y5": REDUCED_INITIAL["y5"],
    "x7": REDUCED_INITIAL["x7"],
    "y7": REDUCED_INITIAL["y7"],
}

TINY_SUPPRESSED_INITIAL = {
    "x1": EQUIVARIANT_INITIAL["x1"],
    "y1": EQUIVARIANT_INITIAL["y1"],
    "x3": EQUIVARIANT_INITIAL["x3"],
    "y3": EQUIVARIANT_INITIAL["y3"],
    "x5": EQUIVARIANT_INITIAL["x5"],
    "y5": EQUIVARIANT_INITIAL["y5"],
}


def exact_basis(k: tuple[int, int, int]) -> tuple[sp.Matrix, sp.Matrix]:
    """Exact version of scripts.gap3.multi_mode_beta_bound.divfree_basis."""
    kvec = sp.Matrix(k)
    n = sum(c * c for c in k)
    sqrt_n = sp.sqrt(n)
    # The code uses v=(1,0,0) if abs(khat[0]) < 0.9, otherwise v=(0,1,0).
    if abs(k[0]) / math.sqrt(n) < 0.9:
        v = sp.Matrix([1, 0, 0])
    else:
        v = sp.Matrix([0, 1, 0])
    raw = v - (sp.Rational(v.dot(kvec), n)) * kvec
    raw_norm = sp.sqrt(sp.simplify(raw.dot(raw)))
    e1 = sp.simplify(raw / raw_norm)
    khat = kvec / sqrt_n
    e2 = sp.simplify(khat.cross(e1))
    return e1, e2


def all_mode_ref(index: int) -> tuple[int, int]:
    n = len(MODES)
    if index < n:
        return index, 1
    return index - n, -1


def enumerate_triads() -> list[tuple[int, int, int, tuple[int, int, int]]]:
    """Triads matching precompute_triads: positive ell, r and s in +/- modes."""
    n = len(MODES)
    doubled = MODES + [(-a, -b, -c) for a, b, c in MODES]
    lookup = {wv: idx for idx, wv in enumerate(doubled)}
    triads = []
    for ell_index, ell in enumerate(MODES):
        for r_index, r in enumerate(doubled):
            s = (ell[0] - r[0], ell[1] - r[1], ell[2] - r[2])
            s_index = lookup.get(s)
            if s_index is not None:
                triads.append((ell_index, r_index, s_index, s))
    return triads


def build_expressions():
    # x0 is fixed to 1 by homogeneity and the observed exact boundary y0=0.
    symbols = sp.symbols("x1 y1 x2 y2 x3 y3 x4 y4 x5 y5 x6 y6 x7 y7 x8 y8", real=True)
    symbol_map = {str(s): s for s in symbols}

    coeffs = [(sp.Integer(1), sp.Integer(0))]
    for i in range(1, 9):
        coeffs.append((symbol_map[f"x{i}"], symbol_map[f"y{i}"]))

    # Fixed phase/sign pattern from the 9-mode optimum.
    I = sp.I
    signed_coeffs = [
        (sp.Integer(1), sp.Integer(0)),
        (-coeffs[1][0], coeffs[1][1]),
        (-coeffs[2][0], coeffs[2][1]),
        (-I * coeffs[3][0], I * coeffs[3][1]),
        (-I * coeffs[4][0], I * coeffs[4][1]),
        (coeffs[5][0], coeffs[5][1]),
        (coeffs[6][0], -coeffs[6][1]),
        (-coeffs[7][0], -coeffs[7][1]),
        (coeffs[8][0], -coeffs[8][1]),
    ]

    bases = [exact_basis(mode) for mode in MODES]
    u_pos = []
    for (c1, c2), (e1, e2) in zip(signed_coeffs, bases):
        u_pos.append(sp.simplify(c1 * e1 + c2 * e2))

    def u(index: int) -> sp.Matrix:
        base, sign = all_mode_ref(index)
        if sign == 1:
            return u_pos[base]
        return sp.conjugate(u_pos[base])

    total = 0
    for ell_index, r_index, s_index, s_vec in enumerate_triads():
        ell2 = sum(c * c for c in MODES[ell_index])
        s_mat = sp.Matrix(s_vec)
        s_dot_ur = (s_mat.T * u(r_index))[0]
        uell_dot_us = (sp.conjugate(u(ell_index)).T * u(s_index))[0]
        total += ell2 * s_dot_ur * uell_dot_us
    # enumerate_triads uses positive ell only.  The negative-output triads are
    # conjugate partners and contribute the same real B contribution for real
    # fields, matching the factor-2 convention in shell_decomp._eval_B_X2_D2.
    B = sp.simplify(-2 * sp.expand(total).coeff(I))

    X2 = 0
    D2 = 0
    for mode, (c1, c2) in zip(MODES, signed_coeffs):
        shell = sum(c * c for c in mode)
        amp2 = sp.simplify(c1 * sp.conjugate(c1) + c2 * sp.conjugate(c2))
        X2 += 2 * shell * amp2
        D2 += 2 * shell * shell * amp2
    return symbols, sp.factor(B), sp.factor(X2), sp.factor(D2)


def stationarity_equations(symbols, B, X2, D2):
    return [
        sp.diff(B, var) * 2 * X2 * D2
        - 2 * B * sp.diff(X2, var) * D2
        - B * X2 * sp.diff(D2, var)
        for var in symbols
    ]


def solve_system(label, symbols, B, X2, D2, guess_map):
    print(f"\n{label}")
    print("-" * len(label))
    print(f"variables: {len(symbols)}")
    print(f"B operation count: {sp.count_ops(B)}")
    print(f"X2 = {sp.factor(X2)}")
    print(f"D2 = {sp.factor(D2)}")

    equations = stationarity_equations(symbols, B, X2, D2)
    guess = [guess_map[str(var)] for var in symbols]
    print("Solving stationarity equations with nsolve (80 digits) ...")
    sol = sp.nsolve(equations, symbols, guess, tol=sp.Float("1e-60"), maxsteps=100, prec=90)
    subs = dict(zip(symbols, sol))
    Bv = sp.N(B.subs(subs), 80)
    Xv = sp.N(X2.subs(subs), 80)
    Dv = sp.N(D2.subs(subs), 80)
    Cv = sp.N(Bv / (Xv * sp.sqrt(Dv)), 80)
    print(f"C = {Cv}")
    print("variables:")
    for var, val in zip(symbols, sol):
        print(f"  {var} = {sp.N(val, 50)}")
    return Cv, sol


def build_reduced_expressions(symbols, B, X2, D2):
    symbol_map = {str(symbol): symbol for symbol in symbols}
    reduction = {
        symbol_map["x2"]: symbol_map["y1"],
        symbol_map["y2"]: symbol_map["x1"],
        symbol_map["x6"]: -symbol_map["x5"],
        symbol_map["y6"]: -symbol_map["y5"],
    }
    reduced_symbols = tuple(
        symbol_map[name]
        for name in ("x1", "y1", "x3", "y3", "x4", "y4", "x5", "y5", "x7", "y7", "x8", "y8")
    )
    return (
        reduced_symbols,
        sp.factor(B.subs(reduction)),
        sp.factor(X2.subs(reduction)),
        sp.factor(D2.subs(reduction)),
    )


def build_equivariant_expressions(symbols, B, X2, D2):
    """Reduce by the exact coordinate-swap equivariance.

    The coordinate swap (a,b,c) -> (b,a,c) pairs modes 1<->2, 3<->4,
    5<->6, and 7<->-8.  In the divergence-free bases used by the code,
    the nontrivial pair action is the orthogonal matrix

        [[sqrt(130)/65,  3*sqrt(455)/65],
         [3*sqrt(455)/65, -sqrt(130)/65]]

    up to the fixed phase signs in build_expressions().
    """
    symbol_map = {str(symbol): symbol for symbol in symbols}
    s = sp.sqrt(130) / 65
    t = 3 * sp.sqrt(455) / 65
    reduction = {
        symbol_map["x2"]: symbol_map["y1"],
        symbol_map["y2"]: symbol_map["x1"],
        symbol_map["x4"]: s * symbol_map["x3"] + t * symbol_map["y3"],
        symbol_map["y4"]: t * symbol_map["x3"] - s * symbol_map["y3"],
        symbol_map["x6"]: -symbol_map["x5"],
        symbol_map["y6"]: -symbol_map["y5"],
        symbol_map["x8"]: -s * symbol_map["x7"] + t * symbol_map["y7"],
        symbol_map["y8"]: t * symbol_map["x7"] + s * symbol_map["y7"],
    }
    equivariant_symbols = tuple(symbol_map[name] for name in ("x1", "y1", "x3", "y3", "x5", "y5", "x7", "y7"))
    return (
        equivariant_symbols,
        sp.factor(B.subs(reduction)),
        sp.factor(X2.subs(reduction)),
        sp.factor(D2.subs(reduction)),
    )


def build_tiny_suppressed_expressions(equivariant_symbols, B, X2, D2):
    """Set the tiny A-shell pair to zero and keep the remaining 6 variables."""
    symbol_map = {str(symbol): symbol for symbol in equivariant_symbols}
    reduction = {
        symbol_map["x7"]: sp.Integer(0),
        symbol_map["y7"]: sp.Integer(0),
    }
    tiny_symbols = tuple(symbol_map[name] for name in ("x1", "y1", "x3", "y3", "x5", "y5"))
    return (
        tiny_symbols,
        sp.factor(B.subs(reduction)),
        sp.factor(X2.subs(reduction)),
        sp.factor(D2.subs(reduction)),
    )


def ratio_grad(symbols, B, X2, D2):
    C_expr = B / (X2 * sp.sqrt(D2))
    return [sp.diff(C_expr, var) for var in symbols]


def polar_diagnostics(symbols, sol):
    print("\nPolar diagnostics for 8-variable solution")
    print("-----------------------------------------")
    sol_map = {str(var): float(sp.N(value, 30)) for var, value in zip(symbols, sol)}
    pairs = [
        ("C_hi", "x1", "y1"),
        ("A_hi", "x3", "y3"),
        ("C_lo", "x5", "y5"),
        ("A_tiny", "x7", "y7"),
    ]
    constants = [sp.sqrt(n) for n in (2, 3, 5, 7, 10, 13, 14, 26, 35, 65, 70, 91, 130, 182, 455, 910)]
    for label, x_name, y_name in pairs:
        x_val = sol_map[x_name]
        y_val = sol_map[y_name]
        radius2 = x_val * x_val + y_val * y_val
        theta = math.atan2(y_val, x_val)
        if theta < 0:
            theta += 2 * math.pi
        theta_over_pi = theta / math.pi
        print(f"  {label:7s}: r^2={radius2:.18g}  theta/pi={theta_over_pi:.18g}  theta={math.degrees(theta):.12f} deg")
        print(f"           nsimplify(r^2)={sp.nsimplify(radius2, constants, full=True, tolerance=1e-14)}")
        print(f"           nsimplify(theta/pi)={sp.nsimplify(theta_over_pi, constants, full=True, tolerance=1e-14)}")


def main() -> None:
    symbols, B, X2, D2 = build_expressions()
    print("Exact k=3 9-mode stationarity system")
    print("=====================================")
    print(f"modes: {len(MODES)} positive")
    print(f"positive-output triads: {len(enumerate_triads())}  (full real-field count: {2 * len(enumerate_triads())})")

    Cv, _ = solve_system("Full phase-fixed system", symbols, B, X2, D2, INITIAL)

    reduced_symbols, B_red, X2_red, D2_red = build_reduced_expressions(symbols, B, X2, D2)
    Cv_red, _ = solve_system(
        "Coordinate-symmetry reduced system",
        reduced_symbols,
        B_red,
        X2_red,
        D2_red,
        REDUCED_INITIAL,
    )
    print(f"\nReduction check: C_full - C_reduced = {sp.N(Cv - Cv_red, 30)}")

    equivariant_symbols, B_eq, X2_eq, D2_eq = build_equivariant_expressions(symbols, B, X2, D2)
    Cv_eq, sol_eq = solve_system(
        "Coordinate-equivariant 8-variable system",
        equivariant_symbols,
        B_eq,
        X2_eq,
        D2_eq,
        EQUIVARIANT_INITIAL,
    )
    print(f"\nEquivariant reduction check: C_full - C_equivariant = {sp.N(Cv - Cv_eq, 30)}")
    polar_diagnostics(equivariant_symbols, sol_eq)

    tiny_symbols, B_tiny, X2_tiny, D2_tiny = build_tiny_suppressed_expressions(equivariant_symbols, B_eq, X2_eq, D2_eq)
    Cv_tiny, sol_tiny = solve_system(
        "Tiny A-pair suppressed 6-variable system",
        tiny_symbols,
        B_tiny,
        X2_tiny,
        D2_tiny,
        TINY_SUPPRESSED_INITIAL,
    )
    print(f"\nTiny-pair value loss: C_equivariant - C_tiny = {sp.N(Cv_eq - Cv_tiny, 30)}")

    full_grad = ratio_grad(equivariant_symbols, B_eq, X2_eq, D2_eq)
    tiny_subs = {var: val for var, val in zip(tiny_symbols, sol_tiny)}
    tiny_symbol_map = {str(symbol): symbol for symbol in equivariant_symbols}
    tiny_subs[tiny_symbol_map["x7"]] = sp.Integer(0)
    tiny_subs[tiny_symbol_map["y7"]] = sp.Integer(0)
    tiny_release = [sp.N(full_grad[-2].subs(tiny_subs), 40), sp.N(full_grad[-1].subs(tiny_subs), 40)]
    print(f"Released tiny-pair gradient at 6-variable optimum: d/d(x7,y7) = {tiny_release}")

    print("\nPSLQ on nsolve C (low-degree sanity only):")
    mp.mp.dps = 80
    c_mp = mp.mpf(str(Cv))
    for degree in (4, 6, 8, 10):
        rel = mp.pslq([c_mp**i for i in range(degree + 1)], tol=mp.mpf("1e-45"), maxcoeff=10_000)
        print(f"  degree <= {degree}, coeff <= 10000: {rel}")


if __name__ == "__main__":
    main()