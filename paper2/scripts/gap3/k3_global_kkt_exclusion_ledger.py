#!/usr/bin/env python3
"""Assemble the k=3 finite global KKT exclusion ledger."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strata-json", required=True)
    parser.add_argument("--dca-report", required=True)
    parser.add_argument("--kkt-suite", required=True)
    parser.add_argument("--kkt-interval-summary", required=True)
    parser.add_argument("--saddle-cert", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    strata = load(args.strata_json)
    dca = load(args.dca_report)
    suite = load(args.kkt_suite)
    kkt_interval = load(args.kkt_interval_summary)
    saddle = load(args.saddle_cert)

    non_dca_rows = suite["rows"]
    kkt_clusters = sorted(int(row["cluster_index"]) for row in non_dca_rows if int(row["hessian_minus_R"]["negative"]) == 0)
    saddle_clusters = sorted(int(row["cluster_index"]) for row in non_dca_rows if int(row["hessian_minus_R"]["negative"]) > 0)
    interval_clusters = sorted(int(cluster) for cluster in kkt_interval["certified_clusters"])
    saddle_cert_clusters = sorted(int(row["cluster_index"]) for row in saddle["rows"] if row["passes"])
    release_negative = [row for row in non_dca_rows if int(row["full_release"]["negative_count_below_1e-10"]) > 0]
    best_non_dca = max(non_dca_rows, key=lambda row: float(row["support_value"]))

    checks = {
        "dca_rows_orbit_collapse": bool(dca["all_dca_rows_orbit_match"] and dca["all_dca_rows_support_size_9"]),
        "non_dca_release_nonnegative": len(release_negative) == 0,
        "kkt_interval_clusters_match": kkt_clusters == interval_clusters,
        "saddle_clusters_match": saddle_clusters == saddle_cert_clusters,
        "kkt_interval_passes": bool(kkt_interval["passes"]),
        "saddle_interval_passes": bool(saddle["passes"]),
    }
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "finite_k3_shell_cluster_kkt_exclusion_closed_at_interval_audit_level" if all(checks.values()) else "incomplete",
        "target": strata["target"],
        "sources": {
            "strata_json": args.strata_json,
            "dca_report": args.dca_report,
            "kkt_suite": args.kkt_suite,
            "kkt_interval_summary": args.kkt_interval_summary,
            "saddle_cert": args.saddle_cert,
        },
        "checks": checks,
        "passes": bool(all(checks.values())),
        "shell_strata_scanned": len(strata["rows"]),
        "dca_rows": int(dca["dca_rows"]),
        "dca_orbit_support_count": int(dca["active_orbit_support_count"]),
        "non_dca_clusters": int(suite["clusters_processed"]),
        "non_dca_kkt_clusters": kkt_clusters,
        "non_dca_saddle_clusters": saddle_clusters,
        "best_non_dca_support_value": float(best_non_dca["support_value"]),
        "best_non_dca_gap_to_target": float(best_non_dca["support_gap_to_target"]),
        "best_interval_kkt_gap_lower": float(kkt_interval["best_target_gap_lower"]),
        "worst_interval_kkt_hessian_lower": float(kkt_interval["worst_hessian_lower"]),
        "worst_saddle_directional_second_upper": float(saddle["worst_directional_second_upper"]),
        "interpretation": (
            "All DCA-containing scanned shell strata collapse to the active orbit. "
            "Every non-DCA value cluster has nonnegative full-block one-mode release coefficients; "
            "clusters with support-local KKT status have interval local certificates below target, "
            "and clusters with negative H(-R) directions have interval negative-curvature certificates."
        ),
        "remaining_formalization": (
            "This closes the finite shell-cluster KKT audit ledger. The manuscript-facing coverage argument is "
            "written in references/K3_GLOBAL_KKT_EXCLUSION.md. Remaining proof-standard work is limited to "
            "exact/rational replacement of floating basis constants if required by the final manuscript standard."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"passes: {report['passes']}")
    print(f"status: {report['status']}")
    print(f"non-DCA KKT clusters: {len(kkt_clusters)} saddle clusters: {len(saddle_clusters)}")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()