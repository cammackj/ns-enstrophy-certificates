#!/usr/bin/env python3
"""Diagnostic probes for the k=11 Route A finite-cell error.

This is not a proof certificate.  It samples the exact k=11 thickened square
for the joint Route A kernel and compares the resulting finite-height envelope
with the current coarse Bernstein constants.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import sympy as sp

try:
    from route_a_joint_kernel_cert import C2, LOG2_HI, route_a_polynomial
except ModuleNotFoundError:  # support importing as scripts.gap3.k11_route_a_error_probe
    from scripts.gap3.route_a_joint_kernel_cert import C2, LOG2_HI, route_a_polynomial


def build_numeric_functions():
    k_joint, _, _, a, b, l_symbol = route_a_polynomial()
    s = a * a
    t = b * b
    ds = sp.diff(k_joint, a) / (2 * a)
    dt = sp.diff(k_joint, b) / (2 * b)

    def g_joint(rho_value: int) -> sp.Expr:
        rho = sp.Rational(rho_value)
        disc = 4 * s * t - (rho - s - t) ** 2
        g_oa = (s - rho) ** 2 * disc**2 / (32 * a**7 * b**7 * rho**2)
        g_pmid = (s - t) ** 2 * disc / (8 * a**3 * b**5 * rho**2)
        return sp.expand(g_oa + g_pmid)

    return {
        "K": sp.lambdify((a, b, l_symbol), k_joint, "numpy"),
        "ds": sp.lambdify((a, b, l_symbol), ds, "numpy"),
        "dt": sp.lambdify((a, b, l_symbol), dt, "numpy"),
        "g1": sp.lambdify((a, b), g_joint(1), "numpy"),
        "g2": sp.lambdify((a, b), g_joint(2), "numpy"),
    }


def finite_height_envelope(
    k: int,
    ds_bound: float,
    dt_bound: float,
    collar_bound: float,
    shrink: float = 1.0,
    radial_shell_scale: bool = False,
) -> tuple[float, float, float, list[tuple[int, float, float, float]]]:
    h = 2.0 ** (-0.5 * k)
    c = math.sqrt(6.0) + 0.75 * h
    k_star_hi = math.sqrt(2.0) * (110.0 * float(LOG2_HI) + 43.0) / 1280.0
    sqrt_k = math.sqrt(k_star_hi)
    threshold = float(C2) * 2.0 ** (0.5 * k)

    rows = []
    zero = 0.0
    upper = 0.0
    for j in range(0, (k - 1) // 2 + 1):
        alpha = 1.0 if j == 0 else float(2**j)
        weight = 1.0 if j == 0 else 2.0 ** (-1.5 * (j - 1))
        if radial_shell_scale:
            perturbation = shrink * h * h * (ds_bound + alpha * alpha * (dt_bound + 2.0 * collar_bound))
        else:
            perturbation = shrink * c * h * (ds_bound + alpha * (dt_bound + 2.0 * collar_bound))
        p_joint = math.sqrt(k_star_hi + perturbation)
        zero += weight * sqrt_k
        upper += weight * p_joint
        rows.append((j, weight * sqrt_k, weight * p_joint, weight * (p_joint - sqrt_k)))
    return threshold, zero, upper, rows


def find_required_shrink(k: int, ds_bound: float, dt_bound: float, collar_bound: float) -> float:
    lo = 0.0
    hi = 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        threshold, _, upper, _ = finite_height_envelope(k, ds_bound, dt_bound, collar_bound, mid)
        if upper <= threshold:
            lo = mid
        else:
            hi = mid
    return lo


def find_required_radial_shrink(k: int, ds_bound: float, dt_bound: float, collar_bound: float) -> float:
    lo = 0.0
    hi = 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        threshold, _, upper, _ = finite_height_envelope(
            k, ds_bound, dt_bound, collar_bound, mid, radial_shell_scale=True
        )
        if upper <= threshold:
            lo = mid
        else:
            hi = mid
    return lo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=401, help="number of sample points in each a,b direction")
    parser.add_argument("--k", type=int, default=11)
    args = parser.parse_args()

    k = args.k
    h = 2.0 ** (-0.5 * k)
    eps = (math.sqrt(6.0) + 0.75 * h) * h
    a_lo = math.sqrt(1.0 - eps)
    a_hi = math.sqrt(2.0 + eps)
    log2 = math.log(2.0)

    funcs = build_numeric_functions()
    grid = np.linspace(a_lo, a_hi, args.grid)
    aa, bb = np.meshgrid(grid, grid, indexing="ij")
    ds_vals = funcs["ds"](aa, bb, log2)
    dt_vals = funcs["dt"](aa, bb, log2)
    g1_vals = funcs["g1"](aa, bb)
    g2_vals = funcs["g2"](aa, bb)
    k_vals = funcs["K"](aa, bb, log2)

    sampled = {
        "abs_ds": float(np.nanmax(np.abs(ds_vals))),
        "abs_dt": float(np.nanmax(np.abs(dt_vals))),
        "collar": float(max(np.nanmax(g1_vals), np.nanmax(g2_vals))),
        "K": float(np.nanmax(k_vals)),
    }

    print(f"k={k}, h={h:.15g}, eps={eps:.15g}")
    print(f"actual thickened a,b interval: [{a_lo:.15g}, {a_hi:.15g}]")
    print(f"grid: {args.grid} x {args.grid}")
    print("sampled exact-thickening suprema")
    for name, value in sampled.items():
        print(f"  {name:8s} {value:.15g}")

    print("\ncoarse certified constants")
    coarse = (0.5, 0.75, 0.4)
    print(f"  abs_ds   {coarse[0]:.15g}")
    print(f"  abs_dt   {coarse[1]:.15g}")
    print(f"  collar   {coarse[2]:.15g}")

    for label, bounds in [("coarse", coarse), ("sampled", (sampled["abs_ds"], sampled["abs_dt"], sampled["collar"]))]:
        threshold, zero, upper, rows = finite_height_envelope(k, *bounds)
        shrink = find_required_shrink(k, *bounds)
        print(f"\n{label} finite-height envelope")
        print(f"  threshold       {threshold:.15g}")
        print(f"  zero-error      {zero:.15g}")
        print(f"  upper           {upper:.15g}")
        print(f"  margin          {threshold - upper:.15g}")
        print(f"  required shrink {shrink:.15g}")
        print("  j  zero_contrib       upper_contrib      loss")
        for j, zero_contrib, upper_contrib, loss in rows:
            print(f"  {j}  {zero_contrib:.12g}  {upper_contrib:.12g}  {loss:.12g}")

    for label, bounds in [("coarse", coarse), ("sampled", (sampled["abs_ds"], sampled["abs_dt"], sampled["collar"]))]:
        threshold, zero, upper, rows = finite_height_envelope(k, *bounds, radial_shell_scale=True)
        shrink = find_required_radial_shrink(k, *bounds)
        print(f"\n{label} radial-shell-scale diagnostic")
        print("  uses h^2 for primitive squared-radius mesh and (2^j h)^2 for relay/rho mesh")
        print(f"  threshold       {threshold:.15g}")
        print(f"  zero-error      {zero:.15g}")
        print(f"  upper           {upper:.15g}")
        print(f"  margin          {threshold - upper:.15g}")
        print(f"  required shrink {shrink:.15g}")
        print("  j  zero_contrib       upper_contrib      loss")
        for j, zero_contrib, upper_contrib, loss in rows:
            print(f"  {j}  {zero_contrib:.12g}  {upper_contrib:.12g}  {loss:.12g}")


if __name__ == "__main__":
    main()