#!/usr/bin/env python3
"""Reduced radical-free stationarity system for the k=3 closed form.

This is the algebraic core after eliminating p, q, and h from the equivariant
k=3 kernel.  It produces polynomial equations in

    t, r, c, s, a, b, e, d, l, w

where c=cos(theta), s=sin(theta), a^2=P_+, b^2=P_-, e^2=Q,
d^2=(a+b)^2+16 r^2 e^2 t^2, l^2=1+8 A_t beta/(alpha G_t), and w=C3^2.

The value equation is

    4480 w (d+a+b) alpha A_t G_t (3+l)^3
      = t^2 (d+3(a+b))^3 (1+l).

The three stationarity equations are log-derivatives in log t, log r, and
theta.  This is the compact system to feed to elimination/root isolation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import sympy as sp

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.gap3.k3_three_variable_reduction import k3_reduced  # noqa: E402


def p_sigma_expr(sigma: int, r_value, c_value, s_value):
    return (
        sp.Integer(1259)
        - 108 * sigma * sp.sqrt(35)
        + 637 * r_value**2
        + 2
        * r_value
        * (
            (162 * sigma * sp.sqrt(7) - 299 * sp.sqrt(5)) * c_value
            - (180 * sp.sqrt(2) + 39 * sigma * sp.sqrt(70)) * s_value
        )
    )


def q_expr(c_value, s_value):
    return 575 + 325 * (c_value**2 - s_value**2) - 300 * sp.sqrt(10) * c_value * s_value


def build_system():
    t, r, c, s, a, b, e, d, ell, w = sp.symbols("t r c s a b e d ell w", real=True)
    variables = (t, r, c, s, a, b, e, d, ell, w)

    p_plus = p_sigma_expr(1, r, c, s)
    p_minus = p_sigma_expr(-1, r, c, s)
    q_value = q_expr(c, s)
    alpha = 2 + 5 * r**2
    beta = 8 + 25 * r**2
    a_t = 5 + 7 * t**2
    g_t = 25 + 49 * t**2
    sigma = a + b
    d2 = sigma**2 + 16 * r**2 * e**2 * t**2
    ell_den = alpha * g_t
    ell_num = alpha * g_t + 8 * a_t * beta

    constraints = [
        c**2 + s**2 - 1,
        a**2 - p_plus,
        b**2 - p_minus,
        e**2 - q_value,
        d**2 - d2,
        ell**2 * ell_den - ell_num,
        4480 * w * (d + sigma) * alpha * a_t * g_t * (3 + ell) ** 3
        - t**2 * (d + 3 * sigma) ** 3 * (1 + ell),
    ]

    def op_log_t(expr):
        return sp.simplify(t * sp.diff(expr, t))

    def op_log_r(expr):
        return sp.simplify(r * sp.diff(expr, r))

    def op_theta(expr):
        return sp.simplify(-s * sp.diff(expr, c) + c * sp.diff(expr, s))

    def stationarity(op):
        p_plus_x = op(p_plus)
        p_minus_x = op(p_minus)
        q_x = op(q_value)
        a_x = p_plus_x / (2 * a)
        b_x = p_minus_x / (2 * b)
        e_x = q_x / (2 * e)
        sigma_x = a_x + b_x
        r2t2 = r**2 * t**2
        d2_x = 2 * sigma * sigma_x + 16 * (op(r2t2) * e**2 + r2t2 * 2 * e * e_x)
        d_x = d2_x / (2 * d)
        ell_num_x = op(ell_num)
        ell_den_x = op(ell_den)
        h_x = (ell_num_x * ell_den - ell_num * ell_den_x) / (ell_den**2)
        ell_x = h_x / (2 * ell)

        expr = (
            2 * op(t) / t
            + 3 * (d_x + 3 * sigma_x) / (d + 3 * sigma)
            + ell_x / (1 + ell)
            - (d_x + sigma_x) / (d + sigma)
            - op(alpha) / alpha
            - op(a_t) / a_t
            - op(g_t) / g_t
            - 3 * ell_x / (3 + ell)
        )
        return sp.factor(sp.together(expr).as_numer_denom()[0])

    equations = constraints + [stationarity(op_log_t), stationarity(op_log_r), stationarity(op_theta)]
    return variables, [sp.factor(sp.expand(equation)) for equation in equations]


def candidate_values() -> dict[str, float]:
    t_value = 0.79944015077205111
    r_value = 0.24853631456081229
    theta = 3.6136991174663304
    c_value = math.cos(theta)
    s_value = math.sin(theta)
    p_plus = float(p_sigma_expr(1, r_value, c_value, s_value).evalf(50))
    p_minus = float(p_sigma_expr(-1, r_value, c_value, s_value).evalf(50))
    q_value = float(q_expr(c_value, s_value).evalf(50))
    a_value = math.sqrt(p_plus)
    b_value = math.sqrt(p_minus)
    e_value = math.sqrt(q_value)
    sigma = a_value + b_value
    d_value = math.sqrt(sigma * sigma + 16 * r_value * r_value * e_value * e_value * t_value * t_value)
    alpha = 2 + 5 * r_value * r_value
    beta = 8 + 25 * r_value * r_value
    a_t = 5 + 7 * t_value * t_value
    g_t = 25 + 49 * t_value * t_value
    ell_value = math.sqrt(1 + 8 * a_t * beta / (alpha * g_t))
    value = k3_reduced(t_value, r_value, theta)
    return {
        "t": t_value,
        "r": r_value,
        "c": c_value,
        "s": s_value,
        "a": a_value,
        "b": b_value,
        "e": e_value,
        "d": d_value,
        "ell": ell_value,
        "w": value * value,
    }


def summarize_system(variables, equations) -> list[dict]:
    rows = []
    for index, equation in enumerate(equations, 1):
        poly = sp.Poly(equation, *variables, extension=[sp.sqrt(2), sp.sqrt(5), sp.sqrt(7)])
        rows.append(
            {
                "equation": index,
                "total_degree": int(poly.total_degree()),
                "terms": int(len(poly.terms())),
                "ops": int(sp.count_ops(equation)),
            }
        )
    return rows


def residuals_at_candidate(variables, equations) -> list[float]:
    values = candidate_values()
    subs = {symbol: values[str(symbol)] for symbol in variables}
    return [float(abs(sp.N(equation.subs(subs), 50))) for equation in equations]


def solve_reduced_system(variables, equations, precision: int) -> tuple[dict[str, str], str, str]:
    values = candidate_values()
    guess = [values[str(variable)] for variable in variables]
    solution = sp.nsolve(
        equations,
        variables,
        guess,
        tol=sp.Float(f"1e-{max(30, precision - 30)}"),
        maxsteps=100,
        prec=precision,
    )
    substitutions = dict(zip(variables, solution))
    max_residual = max(abs(sp.N(equation.subs(substitutions), precision - 10)) for equation in equations)
    root = {str(variable): str(sp.N(value, precision - 20)) for variable, value in zip(variables, solution)}
    c3_value = str(sp.N(sp.sqrt(solution[-1]), precision - 20))
    return root, c3_value, str(sp.N(max_residual, 30))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", default=None)
    parser.add_argument("--nsolve", action="store_true", help="solve the 10-equation system near the k3 root")
    parser.add_argument("--precision", type=int, default=100, help="decimal precision for --nsolve")
    args = parser.parse_args()

    print("k=3 reduced radical-free stationarity system")
    print("============================================")
    variables, equations = build_system()
    rows = summarize_system(variables, equations)
    residuals = residuals_at_candidate(variables, equations)
    print("variables:", ", ".join(str(variable) for variable in variables))
    print(f"equations: {len(equations)}")
    for row, residual in zip(rows, residuals):
        print(
            f"E{row['equation']:02d}: degree={row['total_degree']:2d} "
            f"terms={row['terms']:5d} ops={row['ops']:6d} residual={residual:.3e}"
        )
    print("candidate values:")
    for key, value in candidate_values().items():
        print(f"  {key:>4s} = {value:.17g}")

    root = None
    c3_value = None
    max_residual = None
    if args.nsolve:
        print(f"\nSolving reduced system at precision={args.precision} ...")
        root, c3_value, max_residual = solve_reduced_system(variables, equations, args.precision)
        print(f"C3 = {c3_value}")
        print(f"max residual = {max_residual}")
        for key, value in root.items():
            print(f"  {key:>4s} = {value}")

    if args.save:
        output = Path(args.save)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "variables": [str(variable) for variable in variables],
            "summary": rows,
            "candidate_values": candidate_values(),
            "candidate_residuals_abs": residuals,
            "nsolve_root": root,
            "nsolve_C3": c3_value,
            "nsolve_max_residual_abs": max_residual,
        }
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"saved: {output}")


if __name__ == "__main__":
    main()