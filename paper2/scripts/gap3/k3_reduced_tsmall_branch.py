#!/usr/bin/env python3
"""Interval brancher for the reduced k=3 small-t tail including t=0."""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import mpmath as mp

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.gap3.k3_reduced_interval_branch import (  # noqa: E402
    CANDIDATE_VALUE,
    Box,
    interval_upper,
    parse_bounds,
    parse_ints,
    sqrt_nonnegative,
    split_box,
)


def p_sigma_iv(sigma: int, radius: mp.iv.mpf, theta: mp.iv.mpf) -> mp.iv.mpf:
    return (
        1259
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


def reduced_tsmall_parts_iv(box: Box) -> tuple[mp.iv.mpf, mp.iv.mpf]:
    t_ratio = mp.iv.mpf([box.log_t[0], box.log_t[1]])
    log_r = mp.iv.mpf([box.log_r[0], box.log_r[1]])
    theta = mp.iv.mpf([box.theta[0], box.theta[1]])
    radius = mp.iv.exp(log_r)

    p_plus = p_sigma_iv(1, radius, theta)
    p_minus = p_sigma_iv(-1, radius, theta)
    q_value = q_theta_iv(theta)
    s_gain = sqrt_nonnegative(p_plus) + sqrt_nonnegative(p_minus)
    u_gain = radius * sqrt_nonnegative(2 * q_value)

    d_value = sqrt_nonnegative(s_gain * s_gain + 8 * u_gain * u_gain * t_ratio * t_ratio)
    direction = t_ratio * (d_value + 3 * s_gain) * sqrt_nonnegative((d_value + 3 * s_gain) / (2 * (d_value + s_gain))) / 4
    direction_simple = t_ratio * (s_gain + u_gain * t_ratio)

    alpha = 2 + 5 * radius * radius
    beta = 8 + 25 * radius * radius
    a_t = 5 + 7 * t_ratio * t_ratio
    g_t = 25 + 49 * t_ratio * t_ratio
    ell = sqrt_nonnegative(1 + 8 * a_t * beta / (alpha * g_t))
    scale = sqrt_nonnegative(8 / (alpha * a_t * g_t)) * sqrt_nonnegative(1 + ell) / ((3 + ell) ** mp.iv.mpf(["1.5", "1.5"]))
    factor = sqrt_nonnegative(mp.iv.mpf([70, 70])) * scale / 280
    return factor * direction, factor * direction_simple


def reduced_tsmall_upper(box: Box) -> float:
    exact, simple = reduced_tsmall_parts_iv(box)
    return min(interval_upper(exact), interval_upper(simple))


def make_initial_boxes(t_bounds: tuple[float, float], log_r_bounds: tuple[float, float], theta_bounds: tuple[float, float], splits: tuple[int, int, int]) -> list[Box]:
    axes = [t_bounds, log_r_bounds, theta_bounds]
    grids = []
    for bounds, count in zip(axes, splits):
        lo, hi = bounds
        step = (hi - lo) / count
        grids.append([(lo + index * step, lo + (index + 1) * step) for index in range(count)])
    return [Box(t_box, log_r, theta) for t_box in grids[0] for log_r in grids[1] for theta in grids[2]]


def summarize_box(box: Box, upper: float, depth: int) -> dict:
    return {
        "box_t_log_r_theta": box.as_list(),
        "center_t_log_r_theta": list(box.center()),
        "widths": list(box.widths()),
        "upper": upper,
        "depth": depth,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t-bounds", default=f"0,{math.exp(-2)}")
    parser.add_argument("--log-r-bounds", default="-4,1")
    parser.add_argument("--theta-bounds", default=f"0,{2 * math.pi}")
    parser.add_argument("--initial-splits", default="8,10,40")
    parser.add_argument("--target-margin", type=float, default=1e-5)
    parser.add_argument("--max-boxes", type=int, default=100000)
    parser.add_argument("--max-depth", type=int, default=48)
    parser.add_argument("--dps", type=int, default=50)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    mp.iv.dps = args.dps
    t_bounds = parse_bounds(args.t_bounds)
    if t_bounds[0] < 0:
        raise ValueError("t bounds must be nonnegative")
    log_r_bounds = parse_bounds(args.log_r_bounds)
    theta_bounds = parse_bounds(args.theta_bounds)
    splits = parse_ints(args.initial_splits)
    target = CANDIDATE_VALUE - args.target_margin

    print("k=3 reduced small-t interval branch")
    print("====================================")
    print(f"t={t_bounds} log_r={log_r_bounds} theta={theta_bounds}")
    print(f"initial splits: {splits} target={target:.17g}")
    start_time = time.time()

    heap: list[tuple[float, int, int, Box]] = []
    certified = 0
    failures = 0
    counter = 0
    for box in make_initial_boxes(t_bounds, log_r_bounds, theta_bounds, splits):
        try:
            upper = reduced_tsmall_upper(box)
        except Exception:
            upper = math.inf
            failures += 1
        if upper <= target:
            certified += 1
        else:
            counter += 1
            heapq.heappush(heap, (-upper, counter, 0, box))

    processed = 0
    unresolved: list[tuple[float, int, Box]] = []
    while heap and processed < args.max_boxes:
        if -heap[0][0] <= target:
            certified += len(heap)
            heap.clear()
            break
        neg_upper, _, depth, box = heapq.heappop(heap)
        upper = -neg_upper
        if upper <= target:
            certified += 1
            continue
        if depth >= args.max_depth:
            unresolved.append((upper, depth, box))
            continue
        for child in split_box(box):
            try:
                child_upper = reduced_tsmall_upper(child)
            except Exception:
                child_upper = math.inf
                failures += 1
            if child_upper <= target:
                certified += 1
            else:
                counter += 1
                heapq.heappush(heap, (-child_upper, counter, depth + 1, child))
        processed += 1
        if args.progress_every and processed % args.progress_every == 0:
            print(f"processed={processed} certified={certified} queued={len(heap)} top_upper={-heap[0][0] if heap else None}", flush=True)

    remaining = len(heap)
    while heap and len(unresolved) < max(args.top, 100):
        neg_upper, _, depth, box = heapq.heappop(heap)
        unresolved.append((-neg_upper, depth, box))
    unresolved.sort(key=lambda item: item[0], reverse=True)
    elapsed = time.time() - start_time

    print("\nSummary")
    print("-------")
    print(f"processed boxes: {processed}")
    print(f"certified boxes: {certified}")
    print(f"remaining queued/unresolved total: {remaining}")
    print(f"failures: {failures}")
    if unresolved:
        print(f"top unresolved upper: {unresolved[0][0]:.17g} depth={unresolved[0][1]}")
        print(f"top unresolved box: {unresolved[0][2].as_list()}")
    print(f"elapsed seconds: {elapsed:.1f}")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "target": target,
        "target_margin": args.target_margin,
        "t_bounds": list(t_bounds),
        "log_r_bounds": list(log_r_bounds),
        "theta_bounds": list(theta_bounds),
        "initial_splits": list(splits),
        "processed_boxes": processed,
        "certified_boxes": certified,
        "failures": failures,
        "elapsed_seconds": elapsed,
        "unresolved_total": remaining,
        "unresolved_top": [summarize_box(box, upper, depth) for upper, depth, box in unresolved[: args.top]],
    }
    if not args.no_save:
        output = Path(args.output) if args.output else Path("results") / f"k3_reduced_tsmall_branch_{datetime.now():%Y%m%d_%H%M%S}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved: {output}")


if __name__ == "__main__":
    main()