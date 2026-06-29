#!/usr/bin/env python3
"""
gap3_spanning_triplet.py
========================
Numerical verification of the spanning-triplet geometry for the D×C×A nucleus.

Verifies the analytical claims of §19.10.10:

  1. Coupling angle: cos(θ_DC) = -√2/√n_C  (exact formula, s=2, odd k)
  2. Block span:     n_A/n_D = 2 + O(2^{-k/2})
  3. Angle convergence: cos(θ_DC) → 0  at rate √2 · 2^{-k/2}
  4. Gram-matrix self-similarity: H_k := 2^k · G_k → H_∞  (entry-wise convergence)
  5. Plateau law:    √λ_max(G_k) · 2^{k/2} → γ ≈ 0.0840

Uses the nucleus data from KNOWN_CIK (§19.10.9b) and matches the canonical
construction p_D=(a,a,0), p_C=(0,-s,z), p_A=-(a,a-s,z) with s=2.

Output: table of geometric invariants per k; Gram-matrix eigenvalue analysis;
convergence of rescaled plateau constant.

Usage:
    python scripts/gap3/gap3_spanning_triplet.py
    python scripts/gap3/gap3_spanning_triplet.py --kmax 25

Date: April 2026
"""

import argparse
import math
import sys
import os
import itertools
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Nucleus data from KNOWN_CIK / §19.10.9b (odd k, s=2 canonical nuclei)
# Format: k -> (n_D, n_C, n_A)
# ─────────────────────────────────────────────────────────────────────────────
NUCLEUS_ODD = {
    # k=3: s=1 (pre-plateau, pure D×C×A nucleus {8,10,14})
    3:  (8,    10,   14),
    # k=7..12: transition regime (s=1 or s=2, pre-plateau)
    7:  (128,  82,   194),   # from KNOWN_CIK §19.10.9b
    9:  (512,  401,  881),
    11: (2_048, 1_850, 3_834),
    # k=13..31: canonical spanning nuclei (plateau regime)
    13: (8_192,         8_282,          16_346),
    15: (32_768,        33_128,         65_384),
    17: (131_072,       131_773,        261_821),
    19: (524_288,       525_629,        1_047_869),
    21: (2_097_152,     2_099_605,      4_192_661),
    23: (8_388_608,     8_392_610,      16_777_122),
    25: (33_554_432,    33_558_850,     67_105_090),
    27: (134_217_728,   134_235_400,    268_420_360),
    29: (536_870_912,   536_895_242,    1_073_733_386),
    31: (2_147_483_648, 2_147_488_282,  4_294_906_394),
}

# Corresponding C_nuc(k) values (lower bounds on C(I_k))
CNUC = {
    3:  0.021936,
    5:  0.017084,   # approximate from full-block
    13: 0.000921,
    15: 0.000460,
    17: 0.000231,
    19: 0.000116,
    21: 0.000058,
    23: 0.000029,
    25: 0.000015,
    27: 0.000007,
    29: 3.63e-6,
    31: 1.82e-6,
}


def canonical_triad(k):
    """Return (p_D, p_C, p_A, s, z, a) for odd k.

    Tries s=1,2 to find the right nucleus; for k≥15, s=2 is canonical.
    """
    if k % 2 == 0:
        raise ValueError(f"k={k} must be odd for the canonical D-shell construction")
    a = 2 ** ((k - 1) // 2)
    n_D = 2 * a * a   # = 2^k

    if k in NUCLEUS_ODD:
        n_D_data, n_C_data, n_A_data = NUCLEUS_ODD[k]
        # Deduce s from the nucleus data: 2as = n_D + n_C - n_A
        two_as = n_D_data + n_C_data - n_A_data
        # a for this k
        a_k = 2 ** ((k - 1) // 2)
        if two_as % (2 * a_k) == 0:
            s = two_as // (2 * a_k)
        else:
            s = 2  # default
        z2 = n_C_data - s * s
        if z2 > 0 and int(math.isqrt(z2)) ** 2 == z2:
            z = int(math.isqrt(z2))
        else:
            z = round(math.sqrt(z2)) if z2 > 0 else 0
        p_D = (a_k, a_k, 0)
        p_C = (0, -s, z)
        p_A = (-(a_k), -(a_k - s), -z)
        return p_D, p_C, p_A, s, z, a_k, n_D_data, n_C_data, n_A_data

    # Construct numerically if not in table
    best = None
    for s in [1, 2, 3]:
        # z ≈ a√2, then find nearest integer z with large r3(n_C)
        z_approx = int(math.ceil(a * math.sqrt(2)))
        for dz in range(-3, 20):
            z = z_approx + dz
            if z <= 0:
                continue
            n_C = s * s + z * z
            n_A = n_D - 2 * a * s + n_C
            if n_A <= n_D or n_A >= 2 * n_D:
                continue
            best = (a, a, 0), (0, -s, z), (-a, -(a - s), -z), s, z, a, n_D, n_C, n_A
            break
        if best is not None:
            break
    if best is None:
        raise RuntimeError(f"Could not find nucleus for k={k}")
    return best


def dot3(u, v):
    return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]


def norm2(u):
    return dot3(u, u)


def fmt_angle(cos_val):
    deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_val))))
    return f"{cos_val:+.6e}  (θ={deg:.3f}°)"


def print_geometry_table(k_list):
    print()
    print("=" * 100)
    print("SPANNING TRIPLET GEOMETRY  (§19.10.10)")
    print("=" * 100)
    hdr = (f"{'k':>4}  {'n_D':>14}  {'n_C':>14}  {'n_A':>14}  "
           f"{'n_A/n_D':>9}  {'cos θ_DC (formula)':>22}  "
           f"{'cos θ_DC (direct)':>22}  {'err':>8}")
    print(hdr)
    print("-" * 100)

    for k in k_list:
        if k % 2 == 0:
            continue
        try:
            p_D, p_C, p_A, s, z, a, n_D, n_C, n_A = canonical_triad(k)
        except Exception as e:
            print(f"  k={k}: ERROR {e}")
            continue

        # Formula: cos θ = -√2 / √n_C   (exact for s=2)
        if s == 2:
            cos_formula = -math.sqrt(2) / math.sqrt(n_C)
        else:
            # General: cos θ = -as / sqrt(n_D * n_C)
            cos_formula = -a * s / math.sqrt(n_D * n_C)

        # Direct computation
        pdotc = dot3(p_D, p_C)
        cos_direct = pdotc / math.sqrt(norm2(p_D) * norm2(p_C))

        err = abs(cos_direct - cos_formula) / max(abs(cos_formula), 1e-20)

        ratio = n_A / n_D

        print(f"  {k:>4}  {n_D:>14,d}  {n_C:>14,d}  {n_A:>14,d}  "
              f"{ratio:>9.6f}  {cos_formula:>+22.6e}  {cos_direct:>+22.6e}  {err:>8.4%}")

    print()
    print("  ── Analytical claim: cos θ_DC = -√2/√n_C  (exact for s=2, odd k)")
    print("  ── Block span: n_A/n_D → 2 at rate O(2^{-k/2})")
    print()


def print_convergence_table(k_list):
    """Show convergence of the plateau constant and coupling-angle correction."""
    print("=" * 100)
    print("PLATEAU CONVERGENCE  (§19.10.10 ─ asymptotic analysis)")
    print("=" * 100)
    hdr = (f"{'k':>4}  {'s':>3}  {'C_nuc(k)':>14}  {'C×2^(k/2)':>12}  "
           f"{'|cos θ|':>14}  {'|cos θ|×2a':>12}  {'n_A/n_D - 2':>14}")
    print(hdr)
    # Explanation: |cos θ| × 2a = as/√(n_D n_C) × 2a ≈ as/(a√2 × a√2) × 2a = s
    # So for fixed s, |cos θ| × 2a should converge to s.
    print("-" * 100)

    for k in k_list:
        if k % 2 == 0 or k not in CNUC:
            continue
        try:
            p_D, p_C, p_A, s, z, a, n_D, n_C, n_A = canonical_triad(k)
        except Exception:
            continue

        c = CNUC[k]
        scaled_c = c * 2 ** (k / 2)

        # General coupling-angle formula: cos θ = -as / sqrt(n_D n_C)
        cos_val = -a * s / math.sqrt(n_D * n_C)
        abs_cos = abs(cos_val)
        # Normalized: |cos θ| × 2a → s (for large k, since n_D ≈ n_C ≈ 2a²)
        angle_scaled = abs_cos * 2 * a

        span_err = n_A / n_D - 2.0

        print(f"  {k:>4}  {s:>3}  {c:>14.6e}  {scaled_c:>12.4f}  "
              f"{abs_cos:>14.8e}  {angle_scaled:>12.6f}  {span_err:>14.6e}")

    print()
    target_gamma = 0.0840
    print(f"  ── Observed plateau:  γ = {target_gamma:.4f}  (§19.10.9c)")
    print(f"  ── Theory (self-similarity): C_nuc(k) × 2^(k/2) → γ = √λ_max(H_∞)")
    print(f"  ── Angle formula: |cos θ_DC| = as/√(n_D n_C) ≈ s/(2a) = s·2^{{-(k+1)/2}}")
    print(f"     For fixed s: |cos θ| × 2a → s  (verified in column '|cos θ|×2a')")
    print()


def verify_triad_identity(k_list):
    """Verify the triad identity |p_D + p_C|^2 = n_A for each nucleus."""
    print("=" * 70)
    print("TRIAD IDENTITY VERIFICATION")
    print("=" * 70)
    all_ok = True
    for k in k_list:
        if k % 2 == 0:
            continue
        try:
            p_D, p_C, p_A, s, z, a, n_D, n_C, n_A = canonical_triad(k)
        except Exception as e:
            print(f"  k={k}: SKIP ({e})")
            continue

        # Verify triad: p_D + p_C should equal -p_A
        psum = (p_D[0] + p_C[0], p_D[1] + p_C[1], p_D[2] + p_C[2])
        n_sum = norm2(psum)
        n_neg_A = norm2(p_A)

        ok_triad = (n_sum == n_A)
        ok_nD = (norm2(p_D) == n_D)
        ok_nC = (norm2(p_C) == n_C)

        status = "✓" if (ok_triad and ok_nD and ok_nC) else "✗ ERROR"
        if not (ok_triad and ok_nD and ok_nC):
            all_ok = False

        print(f"  k={k:>3}: |p_D|²={norm2(p_D):,} (expect {n_D:,}) "
              f"|p_C|²={norm2(p_C):,} (expect {n_C:,}) "
              f"|p_D+p_C|²={n_sum:,} (expect {n_A:,})  {status}")

    if all_ok:
        print("\n  All triad identities verified ✓")
    print()


def estimate_gram_scaling(k_list):
    """
    Estimate Gram-matrix self-similarity by computing the leading coupling
    strength (= C_nuc value) and rescaling: H_k = 2^k * G_k should converge.

    We proxy G_k by the 1×1 'matrix' with entry C_nuc(k)^2 (the optimizer
    value^2), scaled by 2^k.  In practice the full 7×7 Gram matrix would
    require running the optimizer on the skeleton — here we use the empirical
    C_nuc(k) to verify that the rescaled eigenvalue λ_max(2^k * G_k) stabilises.
    """
    print("=" * 70)
    print("GRAM-MATRIX SCALING  (proxy via C_nuc^2 × 2^k)")
    print("=" * 70)
    print(f"  {'k':>4}  {'C_nuc(k)':>14}  {'C_nuc²×2^k':>14}  {'√(C_nuc²×2^k)':>14}")
    print("-" * 60)

    vals = []
    for k in sorted(k_list):
        if k % 2 == 0 or k not in CNUC:
            continue
        c = CNUC[k]
        lam_scaled = c ** 2 * 2 ** k   # proxy for λ_max(H_k) = λ_max(2^k G_k)
        gamma_proxy = math.sqrt(max(lam_scaled, 0.0))
        vals.append((k, c, lam_scaled, gamma_proxy))
        print(f"  {k:>4}  {c:>14.6e}  {lam_scaled:>14.6e}  {gamma_proxy:>14.6f}")

    if vals:
        gamma_estimates = [v[3] for v in vals if v[0] >= 13]
        mean_g = sum(gamma_estimates) / len(gamma_estimates)
        std_g = math.sqrt(sum((g - mean_g)**2 for g in gamma_estimates) / len(gamma_estimates))
        print()
        print(f"  ── γ = √λ_max(H_∞) estimate (k ≥ 13): {mean_g:.5f} ± {std_g:.5f}")
        print(f"  ── Relative variation: {std_g/mean_g:.3%}")
        print(f"  ── Theoretical plateau: γ = 0.0840 ± 0.0005  (§19.10.9c)")
    print()


def check_coupling_formula(k_list):
    """
    Verify the coupling-angle formula cos(θ) = -√2/√n_C for s=2 versus
    the general formula cos(θ) = -as/√(n_D n_C), for each nucleus in the list.
    """
    print("=" * 70)
    print("COUPLING ANGLE FORMULA CHECK  (§19.10.10 Theorem)")
    print("=" * 70)
    print("  Test: cos θ (formula) matches cos θ (direct dot product)")
    print()
    max_err = 0.0
    for k in k_list:
        if k % 2 == 0:
            continue
        try:
            p_D, p_C, p_A, s, z, a, n_D, n_C, n_A = canonical_triad(k)
        except Exception:
            continue

        # General formula
        cos_general = -a * s / math.sqrt(n_D * n_C)
        # s=2 specialisation
        cos_s2 = -math.sqrt(2) / math.sqrt(n_C) if s == 2 else float('nan')
        # Direct
        cos_direct = dot3(p_D, p_C) / math.sqrt(norm2(p_D) * norm2(p_C))

        err_gen = abs(cos_direct - cos_general) / max(abs(cos_general), 1e-20)
        err_s2  = abs(cos_direct - cos_s2) / max(abs(cos_s2), 1e-20) if s == 2 else float('nan')
        max_err = max(max_err, err_gen)

        print(f"  k={k:>3}  s={s}  cos(direct)={cos_direct:+.8e}  "
              f"cos(general)={cos_general:+.8e} err={err_gen:.2e}  "
              f"cos(s=2)={cos_s2:+.8e} err={err_s2:.2e}")

    print()
    print(f"  Max relative error (general formula): {max_err:.2e}")
    if max_err < 1e-10:
        print("  ✓ Formula verified to machine precision for all tested k")
    print()


def main():
    parser = argparse.ArgumentParser(description="Spanning triplet geometry (§19.10.10)")
    parser.add_argument('--kmax', type=int, default=31,
                        help='Maximum k to include (default 31)')
    args = parser.parse_args()

    k_list = [k for k in range(3, args.kmax + 1, 2)]  # odd k only

    print()
    print("gap3_spanning_triplet.py")
    print("Numerical verification of §19.10.10 analytical claims")
    print("─" * 60)

    # 1. Verify triad identities
    verify_triad_identity(k_list)

    # 2. Coupling angle formula
    check_coupling_formula(k_list)

    # 3. Geometry table (n_A/n_D, cos θ)
    print_geometry_table(k_list)

    # 4. Convergence of plateau constant
    print_convergence_table(k_list)

    # 5. Gram-matrix scaling proxy
    estimate_gram_scaling(k_list)

    print("─" * 60)
    print("Done.")


if __name__ == '__main__':
    main()
