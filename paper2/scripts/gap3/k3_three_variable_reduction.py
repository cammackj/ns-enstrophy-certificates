#!/usr/bin/env python3
"""Verify the explicit p,q,h elimination for the k=3 kernel."""

from __future__ import annotations

import math

from scipy.optimize import minimize


def p_sigma(sigma: int, radius: float, theta: float) -> float:
    return (
        1259
        - 108 * sigma * math.sqrt(35)
        + 637 * radius * radius
        + 2
        * radius
        * (
            (162 * sigma * math.sqrt(7) - 299 * math.sqrt(5)) * math.cos(theta)
            - (180 * math.sqrt(2) + 39 * sigma * math.sqrt(70)) * math.sin(theta)
        )
    )


def q_theta(theta: float) -> float:
    return 575 + 325 * math.cos(2 * theta) - 150 * math.sqrt(10) * math.sin(2 * theta)


def direction_gain(t_ratio: float, radius: float, theta: float) -> float:
    s_gain = math.sqrt(p_sigma(1, radius, theta)) + math.sqrt(p_sigma(-1, radius, theta))
    u_gain = radius * math.sqrt(2 * q_theta(theta))
    if abs(u_gain) < 1e-15:
        return t_ratio * s_gain
    root = math.sqrt(s_gain * s_gain + 8 * u_gain * u_gain * t_ratio * t_ratio)
    sin_phi = (root - s_gain) / (4 * u_gain * t_ratio)
    sin_phi = max(0.0, min(1.0, sin_phi))
    cos_phi = math.sqrt(max(0.0, 1.0 - sin_phi * sin_phi))
    return t_ratio * cos_phi * (s_gain + u_gain * t_ratio * sin_phi)


def scale_gain(t_ratio: float, radius: float) -> tuple[float, float]:
    alpha = 2 + 5 * radius * radius
    beta = 8 + 25 * radius * radius
    a_t = 5 + 7 * t_ratio * t_ratio
    g_t = 25 + 49 * t_ratio * t_ratio
    ell = math.sqrt(1 + 8 * a_t * beta / (alpha * g_t))
    lambda2 = alpha * (1 + ell) / (2 * a_t)
    gain = math.sqrt(8 / (alpha * a_t * g_t)) * math.sqrt(1 + ell) / ((3 + ell) ** 1.5)
    return gain, lambda2


def k3_reduced(t_ratio: float, radius: float, theta: float) -> float:
    gain, _ = scale_gain(t_ratio, radius)
    return math.sqrt(70) * direction_gain(t_ratio, radius, theta) * gain / 280


def recover_pqh(t_ratio: float, radius: float, theta: float) -> tuple[float, float, float]:
    s_gain = math.sqrt(p_sigma(1, radius, theta)) + math.sqrt(p_sigma(-1, radius, theta))
    u_gain = radius * math.sqrt(2 * q_theta(theta))
    if abs(u_gain) < 1e-15:
        sin_phi = 0.0
    else:
        root = math.sqrt(s_gain * s_gain + 8 * u_gain * u_gain * t_ratio * t_ratio)
        sin_phi = (root - s_gain) / (4 * u_gain * t_ratio)
    sin_phi = max(0.0, min(1.0, sin_phi))
    cos_phi = math.sqrt(max(0.0, 1.0 - sin_phi * sin_phi))
    _, lambda2 = scale_gain(t_ratio, radius)
    scale = math.sqrt(lambda2)
    return scale, scale * t_ratio * cos_phi, scale * t_ratio * sin_phi


def main() -> None:
    p0 = 0.6588123322667734
    q0 = 0.5254332985113650
    r0 = 0.2485363066732576
    h0 = 0.0362319980897601
    theta0 = 3.613699113102433
    t0 = math.hypot(q0, h0) / p0
    print("initial (t,r,theta):", repr(t0), repr(r0), repr(theta0))
    print("K3 reduced at initial:", repr(k3_reduced(t0, r0, theta0)))
    print("recovered (p,q,h):", recover_pqh(t0, r0, theta0))

    def objective(vector):
        log_t, log_r, theta = vector
        return -k3_reduced(math.exp(log_t), math.exp(log_r), theta)

    result = minimize(
        objective,
        [math.log(t0), math.log(r0), theta0],
        method="Nelder-Mead",
        options={"maxiter": 20000, "xatol": 1e-14, "fatol": 1e-16},
    )
    t_star = math.exp(result.x[0])
    r_star = math.exp(result.x[1])
    theta_star = result.x[2] % (2 * math.pi)
    print("optimized (t,r,theta):", repr(t_star), repr(r_star), repr(theta_star))
    print("K3 reduced optimized:", repr(k3_reduced(t_star, r_star, theta_star)))
    print("recovered (p,q,h):", recover_pqh(t_star, r_star, theta_star))


if __name__ == "__main__":
    main()