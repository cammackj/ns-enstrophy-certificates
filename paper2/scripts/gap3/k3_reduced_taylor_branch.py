#!/usr/bin/env python3
"""Taylor/gradient interval brancher for the reduced k=3 kernel.

The older interval brancher evaluates the whole reduced expression on each box,
which leaves a large dependency overestimate.  This brancher evaluates a first
order Taylor enclosure

    f(B) <= f(c) + sum_i sup_B |d_i f| * radius_i

using interval automatic differentiation for the gradient.  It is still a
proof-shaping script, but it produces the kind of finite ledger that can be
audited and, if needed, ported to rational interval arithmetic.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mpmath as mp

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.gap3.k3_reduced_interval_branch import (  # noqa: E402
    CANDIDATE_VALUE,
    CANDIDATE_VECTOR,
    Box,
    box_inside,
    box_intersects,
    candidate_box,
    interval_lower,
    interval_upper,
    make_initial_boxes_with_theta,
    parse_bounds,
    parse_ints,
    reduced_iv,
    split_box,
)
from scripts.gap3.k3_three_variable_reduction import k3_reduced  # noqa: E402


TRACK_HESSIAN = False


def iv_const(value: float | int | str) -> mp.iv.mpf:
    text = str(value)
    return mp.iv.mpf([text, text])


def abs_upper(value: mp.iv.mpf) -> float:
    return max(abs(interval_lower(value)), abs(interval_upper(value)))


@dataclass
class AD:
    value: mp.iv.mpf
    grad: tuple[mp.iv.mpf, mp.iv.mpf, mp.iv.mpf]
    hess: tuple[tuple[mp.iv.mpf, mp.iv.mpf, mp.iv.mpf], tuple[mp.iv.mpf, mp.iv.mpf, mp.iv.mpf], tuple[mp.iv.mpf, mp.iv.mpf, mp.iv.mpf]] | None

    @staticmethod
    def const(value: float | int | str) -> "AD":
        zero = iv_const(0)
        hess = ((zero, zero, zero), (zero, zero, zero), (zero, zero, zero)) if TRACK_HESSIAN else None
        return AD(iv_const(value), (zero, zero, zero), hess)

    def __add__(self, other):
        other = ensure_ad(other)
        hess = None
        if self.hess is not None and other.hess is not None:
            hess = tuple(tuple(self.hess[i][j] + other.hess[i][j] for j in range(3)) for i in range(3))
        return AD(
            self.value + other.value,
            tuple(a + b for a, b in zip(self.grad, other.grad)),
            hess,
        )

    __radd__ = __add__

    def __sub__(self, other):
        other = ensure_ad(other)
        hess = None
        if self.hess is not None and other.hess is not None:
            hess = tuple(tuple(self.hess[i][j] - other.hess[i][j] for j in range(3)) for i in range(3))
        return AD(
            self.value - other.value,
            tuple(a - b for a, b in zip(self.grad, other.grad)),
            hess,
        )

    def __rsub__(self, other):
        other = ensure_ad(other)
        return other.__sub__(self)

    def __neg__(self):
        hess = None if self.hess is None else tuple(tuple(-self.hess[i][j] for j in range(3)) for i in range(3))
        return AD(-self.value, tuple(-item for item in self.grad), hess)

    def __mul__(self, other):
        other = ensure_ad(other)
        hess = None
        if self.hess is not None and other.hess is not None:
            rows = []
            for i in range(3):
                row = []
                for j in range(3):
                    row.append(
                        self.value * other.hess[i][j]
                        + other.value * self.hess[i][j]
                        + self.grad[i] * other.grad[j]
                        + other.grad[i] * self.grad[j]
                    )
                rows.append(tuple(row))
            hess = tuple(rows)
        return AD(
            self.value * other.value,
            tuple(self.value * b + other.value * a for a, b in zip(self.grad, other.grad)),
            hess,
        )

    __rmul__ = __mul__

    def reciprocal(self):
        value = 1 / self.value
        first = -1 / (self.value * self.value)
        second = 2 / (self.value * self.value * self.value)
        return unary_ad(self, value, first, second)

    def __truediv__(self, other):
        other = ensure_ad(other)
        return self * other.reciprocal()

    def __rtruediv__(self, other):
        other = ensure_ad(other)
        return other.__truediv__(self)


def ensure_ad(value) -> AD:
    if isinstance(value, AD):
        return value
    return AD.const(value)


def unary_ad(value: AD, out: mp.iv.mpf, first: mp.iv.mpf, second: mp.iv.mpf) -> AD:
    hess = None
    if value.hess is not None:
        rows = []
        for i in range(3):
            row = []
            for j in range(3):
                row.append(first * value.hess[i][j] + second * value.grad[i] * value.grad[j])
            rows.append(tuple(row))
        hess = tuple(rows)
    return AD(out, tuple(first * item for item in value.grad), hess)


def ad_exp(value: AD) -> AD:
    out = mp.iv.exp(value.value)
    return unary_ad(value, out, out, out)


def ad_sin(value: AD) -> AD:
    out = mp.iv.sin(value.value)
    factor = mp.iv.cos(value.value)
    return unary_ad(value, out, factor, -out)


def ad_cos(value: AD) -> AD:
    out = mp.iv.cos(value.value)
    factor = -mp.iv.sin(value.value)
    return unary_ad(value, out, factor, -out)


def ad_sqrt(value: AD) -> AD:
    lo = max(0.0, interval_lower(value.value))
    hi = max(0.0, interval_upper(value.value))
    out = mp.iv.sqrt(mp.iv.mpf([lo, hi]))
    first = 1 / (2 * out)
    second = -1 / (4 * out * out * out)
    return unary_ad(value, out, first, second)


def ad_pow_const(value: AD, exponent: float) -> AD:
    out = value.value ** iv_const(str(exponent))
    factor = iv_const(str(exponent)) * (value.value ** iv_const(str(exponent - 1.0)))
    second = iv_const(str(exponent * (exponent - 1.0))) * (value.value ** iv_const(str(exponent - 2.0)))
    return unary_ad(value, out, factor, second)


def variables_for_box(box: Box) -> tuple[AD, AD, AD]:
    intervals = [box.log_t, box.log_r, box.theta]
    variables = []
    for index, bounds in enumerate(intervals):
        grad = [iv_const(0), iv_const(0), iv_const(0)]
        grad[index] = iv_const(1)
        zero = iv_const(0)
        hess = ((zero, zero, zero), (zero, zero, zero), (zero, zero, zero)) if TRACK_HESSIAN else None
        variables.append(AD(mp.iv.mpf([bounds[0], bounds[1]]), tuple(grad), hess))
    return variables[0], variables[1], variables[2]


def p_sigma_ad(sigma: int, radius: AD, theta: AD) -> AD:
    return (
        AD.const(1259)
        - AD.const(108 * sigma) * AD.const(mp.sqrt(35))
        + AD.const(637) * radius * radius
        + AD.const(2)
        * radius
        * (
            (AD.const(162 * sigma) * AD.const(mp.sqrt(7)) - AD.const(299) * AD.const(mp.sqrt(5))) * ad_cos(theta)
            - (AD.const(180) * AD.const(mp.sqrt(2)) + AD.const(39 * sigma) * AD.const(mp.sqrt(70))) * ad_sin(theta)
        )
    )


def q_theta_ad(theta: AD) -> AD:
    return AD.const(575) + AD.const(325) * ad_cos(AD.const(2) * theta) - AD.const(150) * AD.const(mp.sqrt(10)) * ad_sin(AD.const(2) * theta)


def reduced_ad(box: Box, track_hessian: bool = False) -> AD:
    global TRACK_HESSIAN
    old_track_hessian = TRACK_HESSIAN
    TRACK_HESSIAN = track_hessian
    try:
        return _reduced_ad_impl(box)
    finally:
        TRACK_HESSIAN = old_track_hessian


def _reduced_ad_impl(box: Box) -> AD:
    log_t, log_r, theta = variables_for_box(box)
    t_ratio = ad_exp(log_t)
    radius = ad_exp(log_r)

    p_plus = p_sigma_ad(1, radius, theta)
    p_minus = p_sigma_ad(-1, radius, theta)
    q_value = q_theta_ad(theta)
    s_gain = ad_sqrt(p_plus) + ad_sqrt(p_minus)
    u_gain = radius * ad_sqrt(AD.const(2) * q_value)
    u_scaled = u_gain * t_ratio
    disc = ad_sqrt(s_gain * s_gain + AD.const(8) * u_scaled * u_scaled)
    direction = t_ratio * (disc + AD.const(3) * s_gain) * ad_sqrt((disc + AD.const(3) * s_gain) / (AD.const(2) * (disc + s_gain))) / AD.const(4)

    alpha = AD.const(2) + AD.const(5) * radius * radius
    beta = AD.const(8) + AD.const(25) * radius * radius
    a_t = AD.const(5) + AD.const(7) * t_ratio * t_ratio
    g_t = AD.const(25) + AD.const(49) * t_ratio * t_ratio
    ell = ad_sqrt(AD.const(1) + AD.const(8) * a_t * beta / (alpha * g_t))
    scale = ad_sqrt(AD.const(8) / (alpha * a_t * g_t)) * ad_sqrt(AD.const(1) + ell) / ad_pow_const(AD.const(3) + ell, 1.5)
    return AD.const(mp.sqrt(70)) * direction * scale / AD.const(280)


def center_value(box: Box) -> float:
    log_t, log_r, theta = box.center()
    return float(k3_reduced(math.exp(log_t), math.exp(log_r), theta))


def point_box(center: tuple[float, float, float]) -> Box:
    return Box((center[0], center[0]), (center[1], center[1]), (center[2], center[2]))


def interval_mid(value: mp.iv.mpf) -> float:
    return 0.5 * (interval_lower(value) + interval_upper(value))


def taylor_upper(box: Box, hessian_max_width: float, target: float | None = None) -> tuple[float, float, list[float], float]:
    ad_value = reduced_ad(box, track_hessian=False)
    natural_upper = interval_upper(ad_value.value)
    center = center_value(box)
    radii = [0.5 * width for width in box.widths()]
    grad_abs = [abs_upper(item) for item in ad_value.grad]
    first_order_upper = center + sum(radius * bound for radius, bound in zip(radii, grad_abs))
    upper = min(first_order_upper, natural_upper)
    use_hessian = hessian_max_width > 0 and max(box.widths()) <= hessian_max_width and (target is None or upper > target)
    if use_hessian:
        hessian_ad = reduced_ad(box, track_hessian=True)
        if hessian_ad.hess is None:
            return upper, natural_upper, grad_abs, center
        natural_upper = min(natural_upper, interval_upper(hessian_ad.value))
        upper = min(upper, natural_upper)
        center_ad = reduced_ad(point_box(box.center()), track_hessian=False)
        center_grad = [interval_mid(item) for item in center_ad.grad]
        hess_abs = [[abs_upper(hessian_ad.hess[i][j]) for j in range(3)] for i in range(3)]
        second_order_upper = center + sum(abs(grad) * radius for grad, radius in zip(center_grad, radii))
        second_order_upper += 0.5 * sum(hess_abs[i][j] * radii[i] * radii[j] for i in range(3) for j in range(3))
        upper = min(upper, second_order_upper)
    return upper, natural_upper, grad_abs, center


def summarize_box(box: Box, upper: float, natural_upper: float, center: float, grad_abs: list[float], depth: int) -> dict:
    return {
        "box": box.as_list(),
        "center": list(box.center()),
        "widths": list(box.widths()),
        "upper": upper,
        "natural_upper": natural_upper,
        "center_value": center,
        "grad_abs_upper": grad_abs,
        "depth": depth,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-t-bounds", default="-8,4")
    parser.add_argument("--log-r-bounds", default="-8,3")
    parser.add_argument("--theta-bounds", default=f"0,{2 * math.pi}")
    parser.add_argument("--initial-splits", default="12,12,48")
    parser.add_argument("--candidate-radius", default="0.12,0.12,0.12")
    parser.add_argument("--target-margin", type=float, default=1e-8)
    parser.add_argument("--max-boxes", type=int, default=100000)
    parser.add_argument("--max-depth", type=int, default=42)
    parser.add_argument("--hessian-max-width", type=float, default=0.0)
    parser.add_argument("--dps", type=int, default=50)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    mp.iv.dps = args.dps
    log_t_bounds = parse_bounds(args.log_t_bounds)
    log_r_bounds = parse_bounds(args.log_r_bounds)
    theta_bounds = parse_bounds(args.theta_bounds)
    splits = parse_ints(args.initial_splits)
    radius_values = [float(part.strip()) for part in args.candidate_radius.split(",") if part.strip()]
    if len(radius_values) != 3:
        raise ValueError("--candidate-radius must contain three comma-separated values")
    candidate_neighborhood = candidate_box(radius_values[0], radius_values[1], radius_values[2])
    target = CANDIDATE_VALUE - args.target_margin

    print("k=3 reduced Taylor interval branch")
    print("===================================")
    print(f"bounds: log_t={log_t_bounds} log_r={log_r_bounds} theta={theta_bounds}")
    print(f"initial splits: {splits} target={target:.17g}")
    print(f"candidate neighborhood: {candidate_neighborhood.as_list()}")
    start_time = time.time()

    heap: list[tuple[float, int, int, Box, float, float, list[float], float]] = []
    certified = 0
    candidate_boxes = 0
    interval_failures = 0
    counter = 0

    for box in make_initial_boxes_with_theta(log_t_bounds, log_r_bounds, theta_bounds, splits):
        if box_inside(box, candidate_neighborhood):
            candidate_boxes += 1
            continue
        try:
            upper, natural_upper, grad_abs, center = taylor_upper(box, args.hessian_max_width, target)
        except Exception:
            interval_failures += 1
            try:
                value = reduced_iv(box)
                upper = interval_upper(value)
                natural_upper = upper
                grad_abs = [math.inf, math.inf, math.inf]
                center = center_value(box)
            except Exception:
                upper = math.inf
                natural_upper = math.inf
                grad_abs = [math.inf, math.inf, math.inf]
                center = -math.inf
        if upper <= target:
            certified += 1
        else:
            counter += 1
            heapq.heappush(heap, (-upper, counter, 0, box, natural_upper, center, grad_abs, upper))

    processed = 0
    unresolved: list[tuple[float, int, Box, float, float, list[float]]] = []
    while heap and processed < args.max_boxes:
        top_upper = -heap[0][0]
        if top_upper <= target:
            certified += len(heap)
            heap.clear()
            break
        neg_upper, _, depth, box, natural_upper, center, grad_abs, upper = heapq.heappop(heap)
        upper = -neg_upper
        if upper <= target:
            certified += 1
            continue
        if box_inside(box, candidate_neighborhood):
            candidate_boxes += 1
            continue
        if depth >= args.max_depth:
            unresolved.append((upper, depth, box, natural_upper, center, grad_abs))
            continue
        for child in split_box(box):
            if box_inside(child, candidate_neighborhood):
                candidate_boxes += 1
                continue
            try:
                child_upper, child_natural, child_grad, child_center = taylor_upper(child, args.hessian_max_width, target)
            except Exception:
                interval_failures += 1
                try:
                    value = reduced_iv(child)
                    child_upper = interval_upper(value)
                    child_natural = child_upper
                    child_grad = [math.inf, math.inf, math.inf]
                    child_center = center_value(child)
                except Exception:
                    child_upper = math.inf
                    child_natural = math.inf
                    child_grad = [math.inf, math.inf, math.inf]
                    child_center = -math.inf
            if child_upper <= target:
                certified += 1
            else:
                counter += 1
                heapq.heappush(heap, (-child_upper, counter, depth + 1, child, child_natural, child_center, child_grad, child_upper))
        processed += 1
        if args.progress_every and processed % args.progress_every == 0:
            current_upper = -heap[0][0] if heap else float("-inf")
            print(
                f"processed={processed} certified={certified} candidate={candidate_boxes} "
                f"queued={len(heap)} top_upper={current_upper:.12g}",
                flush=True,
            )

    remaining_queue = len(heap)
    while heap and len(unresolved) < max(args.top, 100):
        neg_upper, _, depth, box, natural_upper, center, grad_abs, upper = heapq.heappop(heap)
        unresolved.append((-neg_upper, depth, box, natural_upper, center, grad_abs))
    unresolved.sort(key=lambda item: item[0], reverse=True)
    elapsed = time.time() - start_time

    print("\nSummary")
    print("-------")
    print(f"processed boxes: {processed}")
    print(f"certified boxes: {certified}")
    print(f"candidate boxes: {candidate_boxes}")
    print(f"remaining queued/unresolved total: {remaining_queue}")
    print(f"interval failures: {interval_failures}")
    if unresolved:
        print(f"top unresolved upper: {unresolved[0][0]:.17g} depth={unresolved[0][1]}")
        print(f"top unresolved box: {unresolved[0][2].as_list()}")
    print(f"elapsed seconds: {elapsed:.1f}")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_value": CANDIDATE_VALUE,
        "candidate_vector": list(CANDIDATE_VECTOR),
        "target": target,
        "target_margin": args.target_margin,
        "bounds": [list(log_t_bounds), list(log_r_bounds), list(theta_bounds)],
        "initial_splits": list(splits),
        "candidate_neighborhood": candidate_neighborhood.as_list(),
        "processed_boxes": processed,
        "certified_boxes": certified,
        "candidate_boxes": candidate_boxes,
        "interval_failures": interval_failures,
        "elapsed_seconds": elapsed,
        "unresolved_total": remaining_queue,
        "unresolved_top": [summarize_box(box, upper, natural, center, grad, depth) for upper, depth, box, natural, center, grad in unresolved[: args.top]],
    }
    if not args.no_save:
        output = Path(args.output) if args.output else Path("results") / f"k3_reduced_taylor_branch_{datetime.now():%Y%m%d_%H%M%S}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved: {output}")


if __name__ == "__main__":
    main()