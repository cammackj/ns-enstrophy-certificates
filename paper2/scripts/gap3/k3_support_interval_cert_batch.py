#!/usr/bin/env python3
"""Parallel batch driver for k3_support_interval_local_cert.py."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any


def resolve_workers(requested: int, task_count: int) -> int:
    if requested == 0:
        return min(max(1, os.cpu_count() or 1), max(1, task_count))
    return min(max(1, requested), max(1, task_count))


def parse_clusters(text: str) -> list[int] | None:
    if not text.strip() or text.strip().lower() == "all":
        return None
    return [int(part) for part in text.split(",") if part.strip()]


def selected_clusters(kkt_json: str, clusters_text: str, kkt_only: bool) -> list[int]:
    data = json.loads(Path(kkt_json).read_text(encoding="utf-8"))
    requested = parse_clusters(clusters_text)
    rows = []
    for row in data["rows"]:
        cluster_index = int(row["cluster_index"])
        if requested is not None and cluster_index not in requested:
            continue
        if kkt_only and int(row["hessian_minus_R"]["negative"]):
            continue
        rows.append(cluster_index)
    return sorted(rows)


def run_task(task: tuple[str, int, float, int, str]) -> dict[str, Any]:
    kkt_json, cluster, radius, dps, output_dir = task
    output = Path(output_dir) / f"k3_support_interval_local_cert_cluster{cluster}_r{radius:g}_dps{dps}_20260531.json"
    command = [
        sys.executable,
        "scripts/gap3/k3_support_interval_local_cert.py",
        "--kkt-json",
        kkt_json,
        "--cluster",
        str(cluster),
        "--radius",
        repr(radius),
        "--dps",
        str(dps),
        "--output",
        str(output),
    ]
    process = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    passed = process.returncode == 0 and "passes=True" in process.stdout
    return {
        "cluster_index": cluster,
        "returncode": process.returncode,
        "passes": passed,
        "output": str(output),
        "stdout_tail": process.stdout.splitlines()[-8:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kkt-json", required=True)
    parser.add_argument("--clusters", default="all")
    parser.add_argument("--kkt-only", action="store_true")
    parser.add_argument("--radius", type=float, default=1e-5)
    parser.add_argument("--dps", type=int, default=60)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    clusters = selected_clusters(args.kkt_json, args.clusters, args.kkt_only)
    workers = resolve_workers(args.workers, len(clusters))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"support interval batch: clusters={clusters} workers={workers} radius={args.radius:g} dps={args.dps}")
    tasks = [(args.kkt_json, cluster, args.radius, args.dps, args.output_dir) for cluster in clusters]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_task, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            status = "PASS" if row["passes"] else "FAIL"
            print(f"cluster {row['cluster_index']:2d}: {status} code={row['returncode']} output={row['output']}", flush=True)
            if not row["passes"]:
                for line in row["stdout_tail"]:
                    print(f"    {line}", flush=True)
    rows.sort(key=lambda item: item["cluster_index"])
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": args.kkt_json,
        "clusters": clusters,
        "kkt_only": bool(args.kkt_only),
        "radius": args.radius,
        "dps": args.dps,
        "workers": workers,
        "passes": bool(rows and all(row["passes"] for row in rows)),
        "pass_count": sum(1 for row in rows if row["passes"]),
        "fail_count": sum(1 for row in rows if not row["passes"]),
        "rows": rows,
    }
    summary = Path(args.summary)
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"summary passes={report['passes']} pass_count={report['pass_count']} fail_count={report['fail_count']}")
    print(f"saved: {summary}")


if __name__ == "__main__":
    main()