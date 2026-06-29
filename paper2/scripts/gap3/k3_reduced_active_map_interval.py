#!/usr/bin/env python3
"""Interval enclosure for the k=3 reduced-to-active coefficient map."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mpmath as mp

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.gap3.k3_reduced_active_map import active_coefficients_from_reduced
from scripts.gap3.k3_reduced_interval_branch import Box, interval_lower, interval_upper, sqrt_nonnegative


DEFAULT_CENTER = (-0.22384360556449978, -1.3921663408045954, 3.6136991131024327)


def iv(value: float | int | str) -> mp.iv.mpf:
    text = str(value)
    return mp.iv.mpf([text, text])


@dataclass(frozen=True)
class IComplex:
    re: mp.iv.mpf
    im: mp.iv.mpf

    def __add__(self, other: "IComplex") -> "IComplex":
        return IComplex(self.re + other.re, self.im + other.im)

    def __sub__(self, other: "IComplex") -> "IComplex":
        return IComplex(self.re - other.re, self.im - other.im)

    def __neg__(self) -> "IComplex":
        return IComplex(-self.re, -self.im)

    def mul(self, other: "IComplex") -> "IComplex":
        return IComplex(self.re * other.re - self.im * other.im, self.re * other.im + self.im * other.re)

    def scale(self, factor: mp.iv.mpf) -> "IComplex":
        return IComplex(self.re * factor, self.im * factor)

    def conj(self) -> "IComplex":
        return IComplex(self.re, -self.im)

    def abs(self) -> mp.iv.mpf:
        return sqrt_nonnegative(self.re * self.re + self.im * self.im)

    def as_bounds(self) -> dict:
        return {
            "re": [interval_lower(self.re), interval_upper(self.re)],
            "im": [interval_lower(self.im), interval_upper(self.im)],
            "re_width": interval_upper(self.re) - interval_lower(self.re),
            "im_width": interval_upper(self.im) - interval_lower(self.im),
        }


def ic(re_value: float, im_value: float = 0.0) -> IComplex:
    return IComplex(iv(re_value), iv(im_value))


def c_abs_upper(value: IComplex) -> float:
    return max(abs(interval_lower(value.re)), abs(interval_upper(value.re)), abs(interval_lower(value.im)), abs(interval_upper(value.im)))


def unit_sqrt(value: IComplex) -> tuple[IComplex, dict]:
    modulus = value.abs()
    real_unit = value.re / modulus
    imag_unit = value.im / modulus
    real_part = sqrt_nonnegative((iv(1) + real_unit) / 2)
    imag_abs = sqrt_nonnegative((iv(1) - real_unit) / 2)
    if interval_lower(imag_unit) > 0:
        sign = 1
        imag_part = imag_abs
    elif interval_upper(imag_unit) < 0:
        sign = -1
        imag_part = -imag_abs
    else:
        sign = 0
        width = max(abs(interval_lower(imag_abs)), abs(interval_upper(imag_abs)))
        imag_part = mp.iv.mpf([-width, width])
    return IComplex(real_part, imag_part), {
        "input_re": [interval_lower(value.re), interval_upper(value.re)],
        "input_im": [interval_lower(value.im), interval_upper(value.im)],
        "unit_im": [interval_lower(imag_unit), interval_upper(imag_unit)],
        "imag_sign": sign,
    }


def p_sigma_iv(sigma: int, radius: mp.iv.mpf, theta: mp.iv.mpf) -> mp.iv.mpf:
    return (
        iv(1259)
        - 108 * sigma * mp.iv.sqrt(35)
        + 637 * radius * radius
        + 2
        * radius
        * (
            (162 * sigma * mp.iv.sqrt(7) - 299 * mp.iv.sqrt(5)) * mp.iv.cos(theta)
            - (180 * mp.iv.sqrt(2) + 39 * sigma * mp.iv.sqrt(70)) * mp.iv.sin(theta)
        )
    )


def q_theta_iv(theta: mp.iv.mpf) -> mp.iv.mpf:
    return iv(575) + 325 * mp.iv.cos(2 * theta) - 150 * mp.iv.sqrt(10) * mp.iv.sin(2 * theta)


def reduced_pqh_iv(log_t: mp.iv.mpf, log_r: mp.iv.mpf, theta: mp.iv.mpf) -> tuple[mp.iv.mpf, mp.iv.mpf, mp.iv.mpf, mp.iv.mpf, mp.iv.mpf]:
    t_ratio = mp.iv.exp(log_t)
    radius = mp.iv.exp(log_r)
    s_gain = sqrt_nonnegative(p_sigma_iv(1, radius, theta)) + sqrt_nonnegative(p_sigma_iv(-1, radius, theta))
    u_gain = radius * sqrt_nonnegative(2 * q_theta_iv(theta))
    root = sqrt_nonnegative(s_gain * s_gain + 8 * u_gain * u_gain * t_ratio * t_ratio)
    sin_phi = (root - s_gain) / (4 * u_gain * t_ratio)
    cos_phi = sqrt_nonnegative(1 - sin_phi * sin_phi)
    alpha = 2 + 5 * radius * radius
    beta = 8 + 25 * radius * radius
    a_t = 5 + 7 * t_ratio * t_ratio
    g_t = 25 + 49 * t_ratio * t_ratio
    ell = sqrt_nonnegative(1 + 8 * a_t * beta / (alpha * g_t))
    lambda2 = alpha * (1 + ell) / (2 * a_t)
    scale = sqrt_nonnegative(lambda2)
    return scale, scale * t_ratio * cos_phi, scale * t_ratio * sin_phi, radius, theta


def constants() -> tuple[IComplex, IComplex, IComplex, IComplex, IComplex, IComplex]:
    root = math.sqrt
    return (
        ic(-33 * root(910) + 525 * root(26) / 2, 56 * root(65) + 90 * root(91)),
        ic(-525 * root(26) / 2 - 33 * root(910), 90 * root(91) - 56 * root(65)),
        ic(-21 * root(130) / 2 + 90 * root(182), -(27 * root(455) + 70 * root(13))),
        ic(21 * root(130) / 2 + 90 * root(182), -70 * root(13) + 27 * root(455)),
        ic(23 * root(455) + 189 * root(13), -69 * root(182) + 63 * root(130) / 2),
        ic(-23 * root(455) + 189 * root(13), -(69 * root(182) + 63 * root(130) / 2)),
    )


def active_coefficients_interval(box: Box) -> tuple[list[IComplex], dict]:
    log_t = mp.iv.mpf([box.log_t[0], box.log_t[1]])
    log_r = mp.iv.mpf([box.log_r[0], box.log_r[1]])
    theta = mp.iv.mpf([box.theta[0], box.theta[1]])
    p_value, q_value, h_value, radius, theta_iv = reduced_pqh_iv(log_t, log_r, theta)
    w_value = IComplex(radius * mp.iv.cos(theta_iv), radius * mp.iv.sin(theta_iv))
    c_uv0, c_uv1, c_uvw1, c_uvw3, c_vwr0, c_vwr2 = constants()

    a_value = c_uv0 + c_uvw1.mul(w_value.conj())
    b_value = c_uv1 + c_uvw3.conj().mul(w_value.conj())
    k_value = a_value.mul(b_value.conj())
    v_phase, v_branch = unit_sqrt(k_value.conj())
    s_value = a_value.mul(v_phase) + b_value.mul(v_phase.conj())
    u_phase = s_value.conj().scale(-1 / s_value.abs())
    l_value = c_vwr0.mul(w_value) + c_vwr2.mul(w_value.conj())
    z_phase = v_phase.mul(l_value).conj().scale(-1 / l_value.abs())

    u_value = u_phase.scale(p_value)
    v_value = v_phase.scale(q_value)
    z_value = z_phase.scale(h_value)

    x1, y1 = u_value.re, u_value.im
    x3, y3 = v_value.re, v_value.im
    x5, y5 = w_value.re, w_value.im
    x7, y7 = z_value.re, z_value.im
    swap_s = iv(math.sqrt(130) / 65)
    swap_t = iv(3 * math.sqrt(455) / 65)
    x2, y2 = y1, x1
    x4, y4 = swap_s * x3 + swap_t * y3, swap_t * x3 - swap_s * y3
    x6, y6 = -x5, -y5
    x8, y8 = -swap_s * x7 + swap_t * y7, swap_t * x7 + swap_s * y7

    coeffs = [
        IComplex(iv(1), iv(0)),
        IComplex(iv(0), iv(0)),
        IComplex(-x1, iv(0)),
        IComplex(y1, iv(0)),
        IComplex(-x2, iv(0)),
        IComplex(y2, iv(0)),
        IComplex(iv(0), -x3),
        IComplex(iv(0), y3),
        IComplex(iv(0), -x4),
        IComplex(iv(0), y4),
        IComplex(-x5, iv(0)),
        IComplex(-y5, iv(0)),
        IComplex(x6, iv(0)),
        IComplex(-y6, iv(0)),
        IComplex(-x7, iv(0)),
        IComplex(-y7, iv(0)),
        IComplex(x8, iv(0)),
        IComplex(-y8, iv(0)),
    ]
    diagnostics = {
        "v_phase_branch": v_branch,
        "u_phase_abs_interval": [interval_lower(u_phase.abs()), interval_upper(u_phase.abs())],
        "v_phase_abs_interval": [interval_lower(v_phase.abs()), interval_upper(v_phase.abs())],
        "z_phase_abs_interval": [interval_lower(z_phase.abs()), interval_upper(z_phase.abs())],
        "p_interval": [interval_lower(p_value), interval_upper(p_value)],
        "q_interval": [interval_lower(q_value), interval_upper(q_value)],
        "h_interval": [interval_lower(h_value), interval_upper(h_value)],
        "radius_interval": [interval_lower(radius), interval_upper(radius)],
    }
    return coeffs, diagnostics


def parse_triple(text: str) -> tuple[float, float, float]:
    parts = [float(part.strip()) for part in text.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("expected three comma-separated values")
    return parts[0], parts[1], parts[2]


def make_box(center: tuple[float, float, float], radius: tuple[float, float, float]) -> Box:
    return Box(
        (center[0] - radius[0], center[0] + radius[0]),
        (center[1] - radius[1], center[1] + radius[1]),
        (center[2] - radius[2], center[2] + radius[2]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--center", default=",".join(repr(item) for item in DEFAULT_CENTER))
    parser.add_argument("--radius", default="0.0002,0.0002,0.0002")
    parser.add_argument("--dps", type=int, default=80)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    mp.iv.dps = args.dps
    center = parse_triple(args.center)
    radius = parse_triple(args.radius)
    box = make_box(center, radius)
    coeffs, diagnostics = active_coefficients_interval(box)
    center_coeffs = active_coefficients_from_reduced(*center)
    rows = []
    contains_center = []
    for index, coeff in enumerate(coeffs):
        center_value = center_coeffs[index]
        row = coeff.as_bounds()
        center_contained = bool(
            interval_lower(coeff.re) <= center_value.real <= interval_upper(coeff.re)
            and interval_lower(coeff.im) <= center_value.imag <= interval_upper(coeff.im)
        )
        row.update(
            {
                "index": index,
                "mode_index": index // 2,
                "basis_coordinate": "e1" if index % 2 == 0 else "e2",
                "center_re": float(center_value.real),
                "center_im": float(center_value.imag),
                "contains_center": center_contained,
            }
        )
        contains_center.append(row["contains_center"])
        rows.append(row)

    max_width = max(max(row["re_width"], row["im_width"]) for row in rows)
    max_center_radius = max(
        max(
            abs(row["center_re"] - row["re"][0]),
            abs(row["center_re"] - row["re"][1]),
            abs(row["center_im"] - row["im"][0]),
            abs(row["center_im"] - row["im"][1]),
        )
        for row in rows
    )
    print("k=3 reduced active map interval enclosure")
    print("==========================================")
    print(f"center={center} radius={radius} dps={args.dps}")
    print(f"all center coefficients contained={all(contains_center)}")
    print(f"max coefficient interval width={max_width:.12e}")
    print(f"max center-to-endpoint radius={max_center_radius:.12e}")
    print(f"v phase branch sign={diagnostics['v_phase_branch']['imag_sign']} unit_im={diagnostics['v_phase_branch']['unit_im']}")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "active_map_interval_enclosure_not_hessian_certificate",
        "center": list(center),
        "radius": list(radius),
        "box": box.as_list(),
        "dps": args.dps,
        "all_center_coefficients_contained": all(contains_center),
        "max_coefficient_interval_width": max_width,
        "max_center_to_endpoint_radius": max_center_radius,
        "diagnostics": diagnostics,
        "coefficient_intervals": rows,
    }
    if not args.no_save:
        output = Path(args.output) if args.output else Path("results") / f"k3_reduced_active_map_interval_{datetime.now():%Y%m%d_%H%M%S}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved: {output}")


if __name__ == "__main__":
    main()