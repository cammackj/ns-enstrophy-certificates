#!/usr/bin/env python3
"""Map the reduced k=3 variables to the 9 active complex coefficients."""

from __future__ import annotations

import cmath
import math

import numpy as np

from scripts.gap3.k3_three_variable_reduction import recover_pqh


def _coefficients() -> tuple[complex, complex, complex, complex, complex, complex]:
    root = math.sqrt
    c_uv0 = -33 * root(910) + 525 * root(26) / 2 + 1j * (56 * root(65) + 90 * root(91))
    c_uv1 = -525 * root(26) / 2 - 33 * root(910) - 1j * (-90 * root(91) + 56 * root(65))
    c_uvw1 = -21 * root(130) / 2 + 90 * root(182) - 1j * (27 * root(455) + 70 * root(13))
    c_uvw3 = 21 * root(130) / 2 + 90 * root(182) + 1j * (-70 * root(13) + 27 * root(455))
    c_vwr0 = 23 * root(455) + 189 * root(13) + 1j * (-69 * root(182) + 63 * root(130) / 2)
    c_vwr2 = -23 * root(455) + 189 * root(13) - 1j * (69 * root(182) + 63 * root(130) / 2)
    return c_uv0, c_uv1, c_uvw1, c_uvw3, c_vwr0, c_vwr2


C_UV0, C_UV1, C_UVW1, C_UVW3, C_VWR0, C_VWR2 = _coefficients()


def equivariant_variables(log_t: float, log_r: float, theta: float) -> tuple[complex, complex, complex, complex]:
    """Return the four equivariant complex variables (u,v,w,z).

    The compressed numerator in ``k3_kernel_compress.py`` is
    ``Phi = -455 B/8``.  Therefore the active maximizer of ``B`` minimizes the
    two phase-linear pieces of ``Phi``; this contributes the explicit minus
    signs below.
    """
    t_ratio = math.exp(log_t)
    radius = math.exp(log_r)
    p_value, q_value, h_value = recover_pqh(t_ratio, radius, theta)
    w_value = radius * cmath.exp(1j * theta)

    a_value = C_UV0 + C_UVW1 * w_value.conjugate()
    b_value = C_UV1 + C_UVW3.conjugate() * w_value.conjugate()
    beta = 0.5 * (cmath.phase(b_value) - cmath.phase(a_value))
    u_phase_sum = a_value * cmath.exp(1j * beta) + b_value * cmath.exp(-1j * beta)
    alpha = -cmath.phase(u_phase_sum) + math.pi

    z_phase_sum = C_VWR0 * w_value + C_VWR2 * w_value.conjugate()
    gamma = -beta - cmath.phase(z_phase_sum) + math.pi
    return (
        p_value * cmath.exp(1j * alpha),
        q_value * cmath.exp(1j * beta),
        w_value,
        h_value * cmath.exp(1j * gamma),
    )


def active_coefficients_from_reduced(log_t: float, log_r: float, theta: float) -> np.ndarray:
    """Return 18 complex divergence-free-basis coefficients for the active modes.

    The coefficient order is the support order used by ``support_from_warm``.
    For each positive representative mode, the two entries are the e1/e2 complex
    coefficients.
    """
    u_value, v_value, w_value, z_value = equivariant_variables(log_t, log_r, theta)
    x1, y1 = u_value.real, u_value.imag
    x3, y3 = v_value.real, v_value.imag
    x5, y5 = w_value.real, w_value.imag
    x7, y7 = z_value.real, z_value.imag

    s_value = math.sqrt(130) / 65
    t_value = 3 * math.sqrt(455) / 65
    x2, y2 = y1, x1
    x4, y4 = s_value * x3 + t_value * y3, t_value * x3 - s_value * y3
    x6, y6 = -x5, -y5
    x8, y8 = -s_value * x7 + t_value * y7, t_value * x7 + s_value * y7

    pairs = [
        (1.0 + 0j, 0.0 + 0j),
        (-x1 + 0j, y1 + 0j),
        (-x2 + 0j, y2 + 0j),
        (-1j * x3, 1j * y3),
        (-1j * x4, 1j * y4),
        # The warm-state support branch differs from the algebra printout in
        # k3_stationary_system.py by the sign symmetry on mode (3,1,0).
        (-x5 + 0j, -y5 + 0j),
        (x6 + 0j, -y6 + 0j),
        (-x7 + 0j, -y7 + 0j),
        (x8 + 0j, -y8 + 0j),
    ]
    coeffs = np.empty(18, dtype=np.complex128)
    for index, (first, second) in enumerate(pairs):
        coeffs[2 * index] = first
        coeffs[2 * index + 1] = second
    return coeffs


def normalized_active_coefficients_from_reduced(log_t: float, log_r: float, theta: float) -> np.ndarray:
    coeffs = active_coefficients_from_reduced(log_t, log_r, theta)
    amplitudes = np.abs(coeffs.reshape(-1, 2)) ** 2
    scale = math.sqrt(float(np.max(np.sum(amplitudes, axis=1))))
    return coeffs / scale