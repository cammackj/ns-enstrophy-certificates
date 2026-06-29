#!/usr/bin/env python3
"""Uniform small-r certificate for the reduced k=3 kernel.

For r <= R the theta dependence can be eliminated by bounding the two
p_sigma radicals and q_theta uniformly.  The remaining one-dimensional bound
in log(t) is then certified by interval subdivision.
"""

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
    interval_lower,
    interval_upper,
    parse_bounds,
)


def iv_const(value: int | float | str) -> mp.iv.mpf:
    text = str(value)
    return mp.iv.mpf([text, text])


def p_cross_norm_iv(sigma: int) -> mp.iv.mpf:
    a_coeff = iv_const(162 * sigma) * mp.iv.sqrt(7) - iv_const(299) * mp.iv.sqrt(5)
    b_coeff = iv_const(180) * mp.iv.sqrt(2) + iv_const(39 * sigma) * mp.iv.sqrt(70)
    return mp.iv.sqrt(a_coeff * a_coeff + b_coeff * b_coeff)


def small_r_bound_iv(log_t_bounds: tuple[float, float], radius_upper: mp.iv.mpf) -> mp.iv.mpf:
    log_t = mp.iv.mpf([log_t_bounds[0], log_t_bounds[1]])
    t_ratio = mp.iv.exp(log_t)
    return small_r_bound_for_t_iv(t_ratio, radius_upper)


def small_r_bound_for_t_iv(t_ratio: mp.iv.mpf, radius_upper: mp.iv.mpf) -> mp.iv.mpf:
    radius2 = radius_upper * radius_upper

    p_plus0 = iv_const(1259) - iv_const(108) * mp.iv.sqrt(35)
    p_minus0 = iv_const(1259) + iv_const(108) * mp.iv.sqrt(35)
    p_plus_upper = p_plus0 + iv_const(637) * radius2 + iv_const(2) * radius_upper * p_cross_norm_iv(1)
    p_minus_upper = p_minus0 + iv_const(637) * radius2 + iv_const(2) * radius_upper * p_cross_norm_iv(-1)
    s_upper = mp.iv.sqrt(p_plus_upper) + mp.iv.sqrt(p_minus_upper)

    q_sqrt_upper = mp.iv.sqrt(iv_const(2300))
    direction_upper = t_ratio * (s_upper + radius_upper * q_sqrt_upper * t_ratio)

    alpha_min = iv_const(2)
    alpha_max = iv_const(2) + iv_const(5) * radius2
    beta_min = iv_const(8)
    a_t = iv_const(5) + iv_const(7) * t_ratio * t_ratio
    g_t = iv_const(25) + iv_const(49) * t_ratio * t_ratio
    ell_lower = mp.iv.sqrt(iv_const(1) + iv_const(8) * a_t * beta_min / (alpha_max * g_t))
    scale_upper = (
        mp.iv.sqrt(iv_const(8) / (alpha_min * a_t * g_t))
        * mp.iv.sqrt(iv_const(1) + ell_lower)
        / ((iv_const(3) + ell_lower) ** iv_const("1.5"))
    )

    return mp.iv.sqrt(70) * direction_upper * scale_upper / iv_const(280)


def small_r_bound_tau_iv(tau_bounds: tuple[float, float], radius_upper: mp.iv.mpf) -> mp.iv.mpf:
    tau = mp.iv.mpf([tau_bounds[0], tau_bounds[1]])
    radius2 = radius_upper * radius_upper

    p_plus0 = iv_const(1259) - iv_const(108) * mp.iv.sqrt(35)
    p_minus0 = iv_const(1259) + iv_const(108) * mp.iv.sqrt(35)
    p_plus_upper = p_plus0 + iv_const(637) * radius2 + iv_const(2) * radius_upper * p_cross_norm_iv(1)
    p_minus_upper = p_minus0 + iv_const(637) * radius2 + iv_const(2) * radius_upper * p_cross_norm_iv(-1)
    s_upper = mp.iv.sqrt(p_plus_upper) + mp.iv.sqrt(p_minus_upper)

    q_sqrt_upper = mp.iv.sqrt(iv_const(2300))
    direction_scaled = s_upper * tau + radius_upper * q_sqrt_upper

    alpha_min = iv_const(2)
    alpha_max = iv_const(2) + iv_const(5) * radius2
    beta_min = iv_const(8)
    tau2 = tau * tau
    a_bar = iv_const(5) * tau2 + iv_const(7)
    g_bar = iv_const(25) * tau2 + iv_const(49)
    ell_lower = mp.iv.sqrt(iv_const(1) + iv_const(8) * a_bar * beta_min / (alpha_max * g_bar))
    scale_scaled = (
        mp.iv.sqrt(iv_const(8) / (alpha_min * a_bar * g_bar))
        * mp.iv.sqrt(iv_const(1) + ell_lower)
        / ((iv_const(3) + ell_lower) ** iv_const("1.5"))
    )
    return mp.iv.sqrt(70) * direction_scaled * scale_scaled / iv_const(280)


def split_interval(bounds: tuple[float, float]) -> tuple[tuple[float, float], tuple[float, float]]:
    lo, hi = bounds
    mid = 0.5 * (lo + hi)
    return (lo, mid), (mid, hi)


def make_initial_intervals(bounds: tuple[float, float], splits: int) -> list[tuple[float, float]]:
    lo, hi = bounds
    step = (hi - lo) / splits
    return [(lo + index * step, lo + (index + 1) * step) for index in range(splits)]


def summarize_interval(bounds: tuple[float, float], upper: float, depth: int) -> dict:
    return {
        "log_t": list(bounds),
        "center_log_t": 0.5 * (bounds[0] + bounds[1]),
        "width": bounds[1] - bounds[0],
        "upper": upper,
        "depth": depth,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-t-bounds", default="-20,50")
    parser.add_argument("--t-bounds", default=None, help="direct t bounds; permits t=0")
    parser.add_argument("--tau-bounds", default=None, help="direct tau=1/t bounds; permits tau=0")
    parser.add_argument("--log-r-bounds", default="-30,-4")
    parser.add_argument("--initial-splits", type=int, default=256)
    parser.add_argument("--target-margin", type=float, default=1e-5)
    parser.add_argument("--max-boxes", type=int, default=100000)
    parser.add_argument("--max-depth", type=int, default=60)
    parser.add_argument("--dps", type=int, default=80)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    mp.iv.dps = args.dps
    direct_mode = None
    direct_bounds = None
    if args.t_bounds is not None and args.tau_bounds is not None:
        raise ValueError("use at most one of --t-bounds and --tau-bounds")
    if args.t_bounds is not None:
        direct_mode = "t"
        direct_bounds = parse_bounds(args.t_bounds)
        if direct_bounds[0] < 0:
            raise ValueError("t bounds must be nonnegative")
        log_t_bounds = direct_bounds
    elif args.tau_bounds is not None:
        direct_mode = "tau"
        direct_bounds = parse_bounds(args.tau_bounds)
        if direct_bounds[0] < 0:
            raise ValueError("tau bounds must be nonnegative")
        log_t_bounds = direct_bounds
    else:
        log_t_bounds = parse_bounds(args.log_t_bounds)
    log_r_bounds = parse_bounds(args.log_r_bounds)
    if log_r_bounds[0] > log_r_bounds[1]:
        raise ValueError("log-r bounds must be ordered")
    radius_upper = mp.iv.exp(mp.iv.mpf([log_r_bounds[1], log_r_bounds[1]]))
    target = CANDIDATE_VALUE - args.target_margin

    print("k=3 reduced small-r uniform certificate")
    print("========================================")
    print(f"log_t={None if direct_mode else log_t_bounds} {direct_mode or 'log'}_bounds={direct_bounds} log_r={log_r_bounds} radius_upper={interval_upper(radius_upper):.17g}")
    print(f"initial splits: {args.initial_splits} target={target:.17g}")
    start_time = time.time()

    heap: list[tuple[float, int, int, tuple[float, float]]] = []
    certified = 0
    failures = 0
    counter = 0
    top_certified_upper = -math.inf
    for interval in make_initial_intervals(log_t_bounds, args.initial_splits):
        try:
            if direct_mode == "t":
                value = small_r_bound_for_t_iv(mp.iv.mpf([interval[0], interval[1]]), radius_upper)
            elif direct_mode == "tau":
                value = small_r_bound_tau_iv(interval, radius_upper)
            else:
                value = small_r_bound_iv(interval, radius_upper)
            upper = interval_upper(value)
        except Exception:
            upper = math.inf
            failures += 1
        if upper <= target:
            certified += 1
            top_certified_upper = max(top_certified_upper, upper)
        else:
            counter += 1
            heapq.heappush(heap, (-upper, counter, 0, interval))

    processed = 0
    unresolved: list[tuple[float, int, tuple[float, float]]] = []
    while heap and processed < args.max_boxes:
        if -heap[0][0] <= target:
            while heap:
                upper = -heapq.heappop(heap)[0]
                certified += 1
                top_certified_upper = max(top_certified_upper, upper)
            break
        neg_upper, _, depth, interval = heapq.heappop(heap)
        upper = -neg_upper
        if upper <= target:
            certified += 1
            top_certified_upper = max(top_certified_upper, upper)
            continue
        if depth >= args.max_depth:
            unresolved.append((upper, depth, interval))
            continue
        for child in split_interval(interval):
            try:
                if direct_mode == "t":
                    child_value = small_r_bound_for_t_iv(mp.iv.mpf([child[0], child[1]]), radius_upper)
                elif direct_mode == "tau":
                    child_value = small_r_bound_tau_iv(child, radius_upper)
                else:
                    child_value = small_r_bound_iv(child, radius_upper)
                child_upper = interval_upper(child_value)
            except Exception:
                child_upper = math.inf
                failures += 1
            if child_upper <= target:
                certified += 1
                top_certified_upper = max(top_certified_upper, child_upper)
            else:
                counter += 1
                heapq.heappush(heap, (-child_upper, counter, depth + 1, child))
        processed += 1
        if args.progress_every and processed % args.progress_every == 0:
            print(
                f"processed={processed} certified={certified} queued={len(heap)} top_upper={-heap[0][0] if heap else None}",
                flush=True,
            )

    remaining = len(heap)
    while heap and len(unresolved) < max(args.top, 100):
        neg_upper, _, depth, interval = heapq.heappop(heap)
        unresolved.append((-neg_upper, depth, interval))
    unresolved.sort(key=lambda item: item[0], reverse=True)
    elapsed = time.time() - start_time

    print("\nSummary")
    print("-------")
    print(f"processed boxes: {processed}")
    print(f"certified intervals: {certified}")
    print(f"remaining queued/unresolved total: {remaining}")
    print(f"failures: {failures}")
    print(f"top certified upper: {top_certified_upper:.17g}")
    if unresolved:
        print(f"top unresolved upper: {unresolved[0][0]:.17g} depth={unresolved[0][1]}")
        print(f"top unresolved log_t interval: {unresolved[0][2]}")
    print(f"elapsed seconds: {elapsed:.1f}")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_value": CANDIDATE_VALUE,
        "target": target,
        "target_margin": args.target_margin,
        "log_t_bounds": None if direct_mode else list(log_t_bounds),
        "direct_mode": direct_mode,
        "direct_bounds": None if direct_bounds is None else list(direct_bounds),
        "log_r_bounds": list(log_r_bounds),
        "radius_upper": interval_upper(radius_upper),
        "initial_splits": args.initial_splits,
        "processed_boxes": processed,
        "certified_intervals": certified,
        "failures": failures,
        "elapsed_seconds": elapsed,
        "unresolved_total": remaining,
        "top_certified_upper": top_certified_upper,
        "unresolved_top": [summarize_interval(interval, upper, depth) for upper, depth, interval in unresolved[: args.top]],
        "bound_description": {
            "p_sigma": "sqrt(P_sigma(0) + 637 R^2 + 2 R sqrt(A_sigma^2+B_sigma^2))",
            "q_theta": "q_theta <= 1150, so u_gain <= R sqrt(2300)",
            "direction": "direction <= t (s_upper + R sqrt(2300) t)",
            "scale": "alpha >= 2, ell >= sqrt(1 + 64 a_t / ((2+5R^2) g_t)), and h(ell)=sqrt(1+ell)/(3+ell)^(3/2) is decreasing",
        },
    }
    if not args.no_save:
        output = Path(args.output) if args.output else Path("results") / f"k3_reduced_rsmall_bound_{datetime.now():%Y%m%d_%H%M%S}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved: {output}")


if __name__ == "__main__":
    main()