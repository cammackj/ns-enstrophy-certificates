#!/usr/bin/env python3
"""Refine unresolved boxes from a reduced k=3 interval-branch run."""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import mpmath as mp

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.gap3.k3_reduced_interval_branch import (  # noqa: E402
    CANDIDATE_VALUE,
    Box,
    box_inside,
    box_intersects,
    candidate_box,
    interval_lower,
    interval_upper,
    reduced_iv,
    split_box,
)
from scripts.gap3.k3_reduced_taylor_branch import taylor_upper  # noqa: E402


def parse_radius(text: str) -> tuple[float, float, float]:
    parts = [float(part.strip()) for part in text.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("expected three comma-separated radius values")
    return parts[0], parts[1], parts[2]


def box_from_json(row: dict) -> Box:
    log_t, log_r, theta = row["box"]
    return Box(tuple(log_t), tuple(log_r), tuple(theta))


def bound_box(box: Box, target: float, method: str, hessian_max_width: float) -> tuple[float, float | None]:
    if method == "taylor":
        try:
            upper, _, _, center = taylor_upper(box, hessian_max_width, target)
            return upper, center
        except Exception:
            pass
    value = reduced_iv(box)
    return interval_upper(value), interval_lower(value)


def refine_box(box: Box, target: float, neighborhood: Box, max_boxes: int, max_depth: int, method: str, hessian_max_width: float) -> dict:
    heap: list[tuple[float, int, int, Box, float]] = []
    certified = 0
    candidate = 0
    failures = 0
    counter = 0

    if box_inside(box, neighborhood):
        return {"status": "candidate", "processed": 0, "certified": 0, "candidate": 1, "failures": 0, "top_upper": None}
    upper, lower = bound_box(box, target, method, hessian_max_width)
    if upper <= target:
        return {"status": "certified", "processed": 0, "certified": 1, "candidate": 0, "failures": 0, "top_upper": upper}
    heapq.heappush(heap, (-upper, counter, 0, box, lower))

    processed = 0
    unresolved = []
    while heap and processed < max_boxes:
        if -heap[0][0] <= target:
            certified += len(heap)
            heap.clear()
            break
        neg_upper, _, depth, current, lower = heapq.heappop(heap)
        upper = -neg_upper
        if upper <= target:
            certified += 1
            continue
        if box_inside(current, neighborhood):
            candidate += 1
            continue
        if depth >= max_depth:
            unresolved.append((upper, depth, current, lower))
            continue
        for child in split_box(current):
            if box_inside(child, neighborhood):
                candidate += 1
                continue
            try:
                child_upper, child_lower = bound_box(child, target, method, hessian_max_width)
            except Exception:
                failures += 1
                child_upper = math.inf
                child_lower = -math.inf
            if child_upper <= target:
                certified += 1
            else:
                counter += 1
                heapq.heappush(heap, (-child_upper, counter, depth + 1, child, child_lower))
        processed += 1

    if heap:
        neg_upper, _, depth, current, lower = heapq.heappop(heap)
        unresolved.append((-neg_upper, depth, current, lower))
    unresolved.sort(key=lambda item: item[0], reverse=True)
    if unresolved:
        top_upper, top_depth, top_box, top_lower = unresolved[0]
        return {
            "status": "unresolved",
            "processed": processed,
            "certified": certified,
            "candidate": candidate,
            "failures": failures,
            "top_upper": top_upper,
            "top_lower": top_lower,
            "top_depth": top_depth,
            "top_box": top_box.as_list(),
        }
    return {
        "status": "certified",
        "processed": processed,
        "certified": certified,
        "candidate": candidate,
        "failures": failures,
        "top_upper": None,
    }


def refine_task(task: tuple[int, dict, float, Box, int, int, int, str, float]) -> dict:
    index, row, target, neighborhood, max_boxes, max_depth, dps, method, hessian_max_width = task
    mp.iv.dps = dps
    original = box_from_json(row)
    result = refine_box(original, target, neighborhood, max_boxes, max_depth, method, hessian_max_width)
    result["input_index"] = index
    result["input_upper"] = row.get("upper")
    result["input_box"] = original.as_list()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--target-margin", type=float, default=1e-5)
    parser.add_argument("--candidate-radius", default="0.15,0.15,0.15")
    parser.add_argument("--max-boxes-per-input", type=int, default=5000)
    parser.add_argument("--max-depth", type=int, default=48)
    parser.add_argument("--dps", type=int, default=50)
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes; 0 uses all CPUs")
    parser.add_argument("--method", choices=["interval", "taylor"], default="interval")
    parser.add_argument("--hessian-max-width", type=float, default=0.0)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    mp.iv.dps = args.dps
    data = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    source_rows = data.get("unresolved_all") or data.get("unresolved_top", [])
    rows = source_rows if args.limit <= 0 else source_rows[: args.limit]
    target = CANDIDATE_VALUE - args.target_margin
    radius = parse_radius(args.candidate_radius)
    neighborhood = candidate_box(*radius)

    print("k=3 reduced interval refine")
    print("============================")
    print(f"input boxes: {len(rows)}  target={target:.17g}")
    start_time = time.time()
    results = []
    tasks = [
        (index, row, target, neighborhood, args.max_boxes_per_input, args.max_depth, args.dps, args.method, args.hessian_max_width)
        for index, row in enumerate(rows, 1)
    ]
    workers = os.cpu_count() if args.workers == 0 else max(1, args.workers)
    if workers == 1 or len(tasks) <= 1:
        for task in tasks:
            result = refine_task(task)
            results.append(result)
            print(
                f"{result['input_index']:02d}/{len(rows)} {result['status']:10s} "
                f"processed={result['processed']:5d} top_upper={result['top_upper']}",
                flush=True,
            )
    else:
        completed = 0
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(refine_task, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed += 1
                print(
                    f"{completed:05d}/{len(rows)} input={result['input_index']:05d} "
                    f"{result['status']:10s} processed={result['processed']:5d} "
                    f"top_upper={result['top_upper']}",
                    flush=True,
                )
    results.sort(key=lambda item: item["input_index"])

    status_counts: dict[str, int] = {}
    for result in results:
        status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1
    elapsed = time.time() - start_time
    print("\nSummary")
    print("-------")
    print(f"status counts: {status_counts}")
    print(f"elapsed seconds: {elapsed:.1f}")
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_json": args.input_json,
        "limit": args.limit,
        "input_rows_available": len(source_rows),
        "input_rows_processed": len(rows),
        "target": target,
        "target_margin": args.target_margin,
        "candidate_neighborhood": neighborhood.as_list(),
        "max_boxes_per_input": args.max_boxes_per_input,
        "max_depth": args.max_depth,
        "method": args.method,
        "hessian_max_width": args.hessian_max_width,
        "status_counts": status_counts,
        "elapsed_seconds": elapsed,
        "results": results,
    }
    if not args.no_save:
        output = Path(args.output) if args.output else Path("results") / f"k3_reduced_interval_refine_{datetime.now():%Y%m%d_%H%M%S}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved: {output}")


if __name__ == "__main__":
    main()