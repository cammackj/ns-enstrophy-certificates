#!/usr/bin/env python3
"""Interval branch-and-bound diagnostic for the reduced k=3 kernel.

This is a proof-shaping tool.  It uses mpmath interval arithmetic to upper-bound
the reduced objective on boxes in (log t, log r, theta).  Boxes inside a small
neighborhood of the known candidate are set aside; the remaining boxes are
branched until their interval upper bound is below a target or the configured
budget is exhausted.

The direction maximization used by k3_three_variable_reduction.py is evaluated
with the equivalent closed form

    max_x sqrt(1-x^2) (s + U x)
      = (D+3s)/4 * sqrt((D+3s)/(2(D+s))),
        D = sqrt(s^2 + 8 U^2),

which is much more interval-friendly than the optimizer's recovered sin(phi)
formula.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mpmath as mp


CANDIDATE_VECTOR = (-0.22384360783153906, -1.3921663090685699, 3.6136991174663304)
CANDIDATE_VALUE = 0.021936469459403762


def parse_bounds(text: str) -> tuple[float, float]:
    left, right = [float(part.strip()) for part in text.split(",", 1)]
    if not left < right:
        raise ValueError(f"invalid bounds {text!r}")
    return left, right


def parse_ints(text: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in text.split(",") if part.strip()]
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise ValueError("expected three positive integers")
    return parts[0], parts[1], parts[2]


def interval_upper(value: mp.iv.mpf) -> float:
    return float(mp.mpf(value.b))


def interval_lower(value: mp.iv.mpf) -> float:
    return float(mp.mpf(value.a))


def interval_width(value: mp.iv.mpf) -> float:
    return interval_upper(value) - interval_lower(value)


def sqrt_nonnegative(value: mp.iv.mpf) -> mp.iv.mpf:
    """Safe sqrt enclosure for radicands known analytically nonnegative."""
    lo = max(0.0, interval_lower(value))
    hi = max(0.0, interval_upper(value))
    return mp.iv.sqrt(mp.iv.mpf([lo, hi]))


@dataclass(frozen=True)
class Box:
    log_t: tuple[float, float]
    log_r: tuple[float, float]
    theta: tuple[float, float]

    def widths(self) -> tuple[float, float, float]:
        return (
            self.log_t[1] - self.log_t[0],
            self.log_r[1] - self.log_r[0],
            self.theta[1] - self.theta[0],
        )

    def center(self) -> tuple[float, float, float]:
        return (
            0.5 * (self.log_t[0] + self.log_t[1]),
            0.5 * (self.log_r[0] + self.log_r[1]),
            0.5 * (self.theta[0] + self.theta[1]),
        )

    def as_list(self) -> list[list[float]]:
        return [list(self.log_t), list(self.log_r), list(self.theta)]


def candidate_box(radius_log_t: float, radius_log_r: float, radius_theta: float) -> Box:
    log_t, log_r, theta = CANDIDATE_VECTOR
    return Box(
        (log_t - radius_log_t, log_t + radius_log_t),
        (log_r - radius_log_r, log_r + radius_log_r),
        (theta - radius_theta, theta + radius_theta),
    )


def contains(container: tuple[float, float], item: tuple[float, float]) -> bool:
    return container[0] <= item[0] and item[1] <= container[1]


def intersects(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return left[0] <= right[1] and right[0] <= left[1]


def box_inside(box: Box, outer: Box) -> bool:
    return contains(outer.log_t, box.log_t) and contains(outer.log_r, box.log_r) and contains(outer.theta, box.theta)


def box_intersects(box: Box, other: Box) -> bool:
    return intersects(box.log_t, other.log_t) and intersects(box.log_r, other.log_r) and intersects(box.theta, other.theta)


def split_box(box: Box) -> list[Box]:
    widths = box.widths()
    axis = max(range(3), key=lambda index: widths[index])
    intervals = [box.log_t, box.log_r, box.theta]
    lo, hi = intervals[axis]
    mid = 0.5 * (lo + hi)
    left = intervals.copy()
    right = intervals.copy()
    left[axis] = (lo, mid)
    right[axis] = (mid, hi)
    return [Box(left[0], left[1], left[2]), Box(right[0], right[1], right[2])]


def p_sigma_iv(sigma: int, radius: mp.iv.mpf, theta: mp.iv.mpf) -> mp.iv.mpf:
    return (
        mp.iv.mpf([1259, 1259])
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
    return 575 + 325 * mp.iv.cos(2 * theta) - 150 * mp.iv.sqrt(10) * mp.iv.sin(2 * theta)


def reduced_iv(box: Box) -> mp.iv.mpf:
    log_t = mp.iv.mpf([box.log_t[0], box.log_t[1]])
    log_r = mp.iv.mpf([box.log_r[0], box.log_r[1]])
    theta = mp.iv.mpf([box.theta[0], box.theta[1]])
    t_ratio = mp.iv.exp(log_t)
    radius = mp.iv.exp(log_r)

    p_plus = p_sigma_iv(1, radius, theta)
    p_minus = p_sigma_iv(-1, radius, theta)
    q_value = q_theta_iv(theta)
    s_gain = sqrt_nonnegative(p_plus) + sqrt_nonnegative(p_minus)
    u_gain = radius * sqrt_nonnegative(2 * q_value)
    u_scaled = u_gain * t_ratio
    disc = sqrt_nonnegative(s_gain * s_gain + 8 * u_scaled * u_scaled)
    direction = t_ratio * (disc + 3 * s_gain) * sqrt_nonnegative((disc + 3 * s_gain) / (2 * (disc + s_gain))) / 4

    alpha = 2 + 5 * radius * radius
    beta = 8 + 25 * radius * radius
    a_t = 5 + 7 * t_ratio * t_ratio
    g_t = 25 + 49 * t_ratio * t_ratio
    ell = sqrt_nonnegative(1 + 8 * a_t * beta / (alpha * g_t))
    scale = sqrt_nonnegative(8 / (alpha * a_t * g_t)) * sqrt_nonnegative(1 + ell) / ((3 + ell) ** mp.iv.mpf(["1.5", "1.5"]))
    return sqrt_nonnegative(mp.iv.mpf([70, 70])) * direction * scale / 280


def make_initial_boxes(log_t_bounds: tuple[float, float], log_r_bounds: tuple[float, float], splits: tuple[int, int, int]) -> list[Box]:
    boxes = []
    theta_bounds = (0.0, 2 * math.pi)
    axes = [log_t_bounds, log_r_bounds, theta_bounds]
    grids = []
    for bounds, count in zip(axes, splits):
        lo, hi = bounds
        step = (hi - lo) / count
        grids.append([(lo + index * step, lo + (index + 1) * step) for index in range(count)])
    for log_t in grids[0]:
        for log_r in grids[1]:
            for theta in grids[2]:
                boxes.append(Box(log_t, log_r, theta))
    return boxes


def make_initial_boxes_with_theta(
    log_t_bounds: tuple[float, float],
    log_r_bounds: tuple[float, float],
    theta_bounds: tuple[float, float],
    splits: tuple[int, int, int],
) -> list[Box]:
    boxes = []
    axes = [log_t_bounds, log_r_bounds, theta_bounds]
    grids = []
    for bounds, count in zip(axes, splits):
        lo, hi = bounds
        step = (hi - lo) / count
        grids.append([(lo + index * step, lo + (index + 1) * step) for index in range(count)])
    for log_t in grids[0]:
        for log_r in grids[1]:
            for theta in grids[2]:
                boxes.append(Box(log_t, log_r, theta))
    return boxes


def summarize_box(box: Box, upper: float, lower: float | None = None) -> dict:
    data = {
        "box": box.as_list(),
        "center": list(box.center()),
        "widths": list(box.widths()),
        "upper": upper,
    }
    if lower is not None:
        data["lower"] = lower
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-t-bounds", default="-8,4")
    parser.add_argument("--log-r-bounds", default="-8,3")
    parser.add_argument("--theta-bounds", default=f"0,{2 * math.pi}")
    parser.add_argument("--initial-splits", default="8,8,24")
    parser.add_argument("--candidate-radius", default="0.08,0.08,0.08")
    parser.add_argument("--target-margin", type=float, default=1e-5)
    parser.add_argument("--max-boxes", type=int, default=25000)
    parser.add_argument("--max-depth", type=int, default=28)
    parser.add_argument("--dps", type=int, default=50)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--save-all-unresolved", action="store_true")
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--output", default=None)
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

    print("k=3 reduced interval branch")
    print("============================")
    print(f"bounds: log_t={log_t_bounds} log_r={log_r_bounds} theta={theta_bounds}")
    print(f"initial splits: {splits}  target={target:.17g}")
    print(f"candidate neighborhood: {candidate_neighborhood.as_list()}")
    start_time = time.time()

    heap: list[tuple[float, int, int, Box, float]] = []
    certified = 0
    candidate_boxes = 0
    interval_failures = 0
    counter = 0

    for box in make_initial_boxes_with_theta(log_t_bounds, log_r_bounds, theta_bounds, splits):
        if box_inside(box, candidate_neighborhood):
            candidate_boxes += 1
            continue
        try:
            value = reduced_iv(box)
            upper = interval_upper(value)
            lower = interval_lower(value)
        except Exception:
            interval_failures += 1
            upper = math.inf
            lower = -math.inf
        if upper <= target and not box_intersects(box, candidate_neighborhood):
            certified += 1
            continue
        counter += 1
        heapq.heappush(heap, (-upper, counter, 0, box, lower))

    processed = 0
    unresolved: list[tuple[float, int, Box, float]] = []
    while heap and processed < args.max_boxes:
        neg_upper, _, depth, box, lower = heapq.heappop(heap)
        upper = -neg_upper
        if upper <= target and not box_intersects(box, candidate_neighborhood):
            certified += 1
            continue
        if box_inside(box, candidate_neighborhood):
            candidate_boxes += 1
            continue
        if depth >= args.max_depth:
            unresolved.append((upper, depth, box, lower))
            continue
        for child in split_box(box):
            if box_inside(child, candidate_neighborhood):
                candidate_boxes += 1
                continue
            try:
                child_value = reduced_iv(child)
                child_upper = interval_upper(child_value)
                child_lower = interval_lower(child_value)
            except Exception:
                interval_failures += 1
                child_upper = math.inf
                child_lower = -math.inf
            if child_upper <= target and not box_intersects(child, candidate_neighborhood):
                certified += 1
            else:
                counter += 1
                heapq.heappush(heap, (-child_upper, counter, depth + 1, child, child_lower))
        processed += 1
        if args.progress_every and processed % args.progress_every == 0:
            current_upper = -heap[0][0] if heap else float("-inf")
            print(
                f"processed={processed} certified={certified} "
                f"candidate={candidate_boxes} queued={len(heap)} top_upper={current_upper:.12g}",
                flush=True,
            )

    remaining_queue = len(heap)
    unresolved_limit = remaining_queue if args.save_all_unresolved else max(args.top, 100)
    while heap and len(unresolved) < unresolved_limit:
        neg_upper, _, depth, box, lower = heapq.heappop(heap)
        unresolved.append((-neg_upper, depth, box, lower))
    unresolved.sort(key=lambda item: item[0], reverse=True)
    elapsed = time.time() - start_time

    print("\nSummary")
    print("-------")
    print(f"processed boxes: {processed}")
    print(f"certified boxes: {certified}")
    print(f"candidate boxes: {candidate_boxes}")
    print(f"remaining queued/unresolved total: {remaining_queue}")
    print(f"remaining queued/unresolved shown: {len(unresolved)}")
    print(f"interval failures: {interval_failures}")
    if unresolved:
        print(f"top unresolved upper: {unresolved[0][0]:.17g} depth={unresolved[0][1]}")
        print(f"top unresolved box: {unresolved[0][2].as_list()}")
    print(f"elapsed seconds: {elapsed:.1f}")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_value": CANDIDATE_VALUE,
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
        "unresolved_count_reported": len(unresolved),
        "unresolved_top": [summarize_box(box, upper, lower) | {"depth": depth} for upper, depth, box, lower in unresolved[: args.top]],
    }
    if args.save_all_unresolved:
        summary["unresolved_all"] = [
            summarize_box(box, upper, lower) | {"depth": depth}
            for upper, depth, box, lower in unresolved
        ]
    if not args.no_save:
        output = Path(args.output) if args.output else Path("results") / f"k3_reduced_interval_branch_{datetime.now():%Y%m%d_%H%M%S}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved: {output}")


if __name__ == "__main__":
    main()