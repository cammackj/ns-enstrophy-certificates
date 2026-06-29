#!/usr/bin/env python3
"""Collect passing k=3 support interval certificates into one ledger."""

from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kkt-json", required=True)
    parser.add_argument("--glob", default="results/k3_support_interval_local_cert_cluster*_*.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    kkt_data = json.loads(Path(args.kkt_json).read_text(encoding="utf-8"))
    kkt_clusters = sorted(int(row["cluster_index"]) for row in kkt_data["rows"] if int(row["hessian_minus_R"]["negative"]) == 0)
    candidates: dict[int, list[dict]] = {cluster: [] for cluster in kkt_clusters}
    for path_text in glob.glob(args.glob):
        path = Path(path_text)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        cluster = int(data.get("cluster_index", -1))
        if cluster not in candidates or not data.get("passes"):
            continue
        candidates[cluster].append(
            {
                "path": str(path),
                "radius": float(data["radius"]),
                "slice_dim": int(data["slice"]["positive_rank"]),
                "hessian_lower": float(data["hessian_positive"]["certified_lower_bound"]),
                "R_critical_upper": float(data["R_critical_upper"]),
                "target_gap_lower": float(data["target_gap_lower_using_text_target"]),
                "slice_polish": bool(data.get("slice_polish", {}).get("enabled", False)),
            }
        )
    selected = []
    missing = []
    for cluster in kkt_clusters:
        if not candidates[cluster]:
            missing.append(cluster)
            continue
        best = max(candidates[cluster], key=lambda item: (item["target_gap_lower"], item["hessian_lower"]))
        selected.append({"cluster_index": cluster, **best})
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": args.kkt_json,
        "expected_kkt_clusters": kkt_clusters,
        "certified_clusters": [row["cluster_index"] for row in selected],
        "missing_clusters": missing,
        "passes": len(missing) == 0,
        "count": len(selected),
        "best_target_gap_lower": min((row["target_gap_lower"] for row in selected), default=None),
        "worst_hessian_lower": min((row["hessian_lower"] for row in selected), default=None),
        "rows": selected,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"expected: {len(kkt_clusters)} certified: {len(selected)} missing: {missing}")
    print(f"passes: {report['passes']}")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()