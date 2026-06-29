#!/usr/bin/env python3
"""Probe closed-form structure for the k=3 DCxA optimiser.

This is a research diagnostic, not a proof.  It fixes the 9 positive modes
seen in the current k=3 warm state, optimises the current certification
objective on that support, evaluates the result with mpmath, and runs
conservative PSLQ checks on the documented high-precision value.
"""

from __future__ import annotations

import math
import os
import sys

import mpmath as mp
import numpy as np
from scipy.optimize import minimize

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.gap3.analysis.inspect_warm_state import (
    extract_amplitudes,
    load_warm_state,
    reconstruct_wavevectors,
)
from scripts.gap3.certify_block_maximum_gpu import _build_mpmath_data, _mpmath_objective
from scripts.gap3.max_b_over_keff import neg_ratio_and_grad
from scripts.gap3.multi_mode_beta_bound import divfree_basis, precompute_triads


# High-precision value already documented in references/NS_CANCELLATION.md
# section 19.10.7.  The optimiser below reproduces this to ~13 digits using
# float64 variables and the current certification objective.
K3_REF = mp.mpf("0.0219364694591953055423450077575")


def build_prob(modes: list[tuple[int, int, int]]) -> dict:
    _, ell_idx, ell2, r_idx, s_idx, s_mat = precompute_triads(modes)
    return {
        "N": len(modes),
        "k2s": np.array([sum(c * c for c in wv) for wv in modes], dtype=float),
        "e1s": np.array([divfree_basis(wv)[0] for wv in modes]),
        "e2s": np.array([divfree_basis(wv)[1] for wv in modes]),
        "ell_idx": ell_idx,
        "ell2": ell2,
        "r_idx": r_idx,
        "s_idx": s_idx,
        "s_mat": s_mat,
        "wavevecs": modes,
    }


def support_from_warm() -> tuple[list[int], list[tuple[int, int, int]], np.ndarray]:
    warm = load_warm_state("results/warm_state/k3_warm.npz")
    wavevecs, hash_ok = reconstruct_wavevectors(warm["k"], warm["wv_hash"])
    if not hash_ok:
        raise RuntimeError("warm-state wavevector hash mismatch")
    x = warm["best_x"]
    amplitudes = extract_amplitudes(x[: 4 * len(wavevecs)])
    support = [i for i in range(len(wavevecs)) if x[4 * i + 3] > -7.999]
    support.sort(key=lambda i: -amplitudes[i])
    modes = [tuple(wavevecs[i]) for i in support]
    x0 = np.concatenate([x[4 * i : 4 * i + 4] for i in support])
    # The ratio is homogeneous.  Removing the largest log-amplitude avoids a
    # spurious boundary at loga=8 while preserving the represented field.
    x0[3::4] -= np.max(x0[3::4])
    return support, modes, x0


def optimise_current_objective(prob: dict, x0: np.ndarray) -> tuple[float, np.ndarray, float]:
    n_modes = prob["N"]
    bounds = [(0.0, math.pi / 2), (0.0, 2 * math.pi), (0.0, 2 * math.pi), (-8.0, 8.0)] * n_modes

    def objective(params: np.ndarray):
        return neg_ratio_and_grad(
            params,
            n_modes,
            prob["e1s"],
            prob["e2s"],
            prob["k2s"],
            prob["ell_idx"],
            prob["ell2"],
            prob["r_idx"],
            prob["s_idx"],
            prob["s_mat"],
        )

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 100_000, "ftol": 1e-16, "gtol": 1e-14},
    )
    return -float(result.fun), result.x, float(np.max(np.abs(result.jac)))


def mpmath_value(prob: dict, params: np.ndarray, dps: int = 80) -> mp.mpf:
    mp.mp.dps = dps
    data = _build_mpmath_data(prob, dps)
    return _mpmath_objective([mp.mpf(str(v)) for v in params], *data)


def pslq_checks() -> None:
    mp.mp.dps = 90
    c = K3_REF
    c2 = c * c
    print("\nConservative PSLQ checks against documented high-precision value:")
    print(f"  C_ref  = {mp.nstr(c, 40)}")
    print(f"  C_ref^2= {mp.nstr(c2, 40)}")

    for degree in (4, 6, 8):
        rel = mp.pslq([c**i for i in range(degree + 1)], tol=mp.mpf("1e-45"), maxcoeff=10_000)
        print(f"  minpoly(C), degree <= {degree}, coeff <= 10000: {rel}")
    for degree in (4, 6, 8):
        rel = mp.pslq([c2**i for i in range(degree + 1)], tol=mp.mpf("1e-45"), maxcoeff=10_000)
        print(f"  minpoly(C^2), degree <= {degree}, coeff <= 10000: {rel}")

    roots = [2, 3, 5, 7, 10, 13, 14, 35, 39, 65, 70, 91, 130, 182]
    simple_surd_basis = [1120 * c2] + [mp.sqrt(n) for n in roots] + [mp.mpf(1)]
    rel = mp.pslq(simple_surd_basis, tol=mp.mpf("1e-45"), maxcoeff=10_000)
    print(f"  1120*C^2 in simple DCxA surd basis, coeff <= 10000: {rel}")


def main() -> None:
    support, modes, x0 = support_from_warm()
    prob = build_prob(modes)
    value_f64, params, grad_max = optimise_current_objective(prob, x0)
    value_mp = mpmath_value(prob, params)

    print("k=3 closed-form probe")
    print("=======================")
    print(f"support indices: {support}")
    print(f"modes ({len(modes)} positive):")
    for index, mode in zip(support, modes):
        shell = sum(c * c for c in mode)
        print(f"  {index:3d}: shell {shell:2d}  {mode}")
    print(f"triads: {len(prob['ell_idx'])}")
    print(f"float64 optimum: {value_f64:.17g}")
    print(f"mpmath at float64 optimum: {mp.nstr(value_mp, 70)}")
    print(f"projected/raw grad max at float64 point: {grad_max:.3e}")

    amplitudes = np.exp(params[3::4])
    max_amp = float(amplitudes.max())
    print("\nrelative amplitude groups:")
    groups = {
        "D": [0],
        "C_hi": [1, 2],
        "A_hi": [3, 4],
        "C_lo": [5, 6],
        "A_tiny": [7, 8],
    }
    for name, indices in groups.items():
        vals = [float(amplitudes[i] / max_amp) for i in indices]
        print(f"  {name:6s}: {vals}  sum={sum(vals):.15g}")

    pslq_checks()


if __name__ == "__main__":
    main()