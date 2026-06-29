#!/usr/bin/env python3
"""Report DCA-containing k=3 shell-stratum rows that collapse to the active orbit."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.gap3.k3_active_set_verify import build_problem_scope  # noqa: E402
from scripts.gap3.k3_penalized_fullblock_scan import active_indices  # noqa: E402


def positive_rep(mode: tuple[int, int, int]) -> tuple[int, int, int]:
    negative = tuple(-component for component in mode)
    return mode if mode > negative else negative


def active_orbit_supports() -> set[frozenset[tuple[int, int, int]]]:
    problem = build_problem_scope("full-block")
    canonical = [tuple(int(component) for component in problem["wavevecs"][index]) for index in active_indices(problem)]
    supports: set[frozenset[tuple[int, int, int]]] = set()
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            image = []
            for mode in canonical:
                mapped = tuple(signs[index] * mode[perm[index]] for index in range(3))
                image.append(positive_rep(mapped))
            supports.add(frozenset(image))
    return supports


def significant_support(row: dict, cutoff: float) -> tuple[frozenset[tuple[int, int, int]], list[dict]]:
    params = np.asarray(row["best_params"], dtype=float)
    wavevectors = [tuple(int(component) for component in mode) for mode in row["wavevectors"]]
    k2s = np.asarray([sum(component * component for component in mode) for mode in wavevectors], dtype=float)
    weights = 2.0 * k2s * np.exp(params[3::4])
    weights /= float(np.sum(weights))
    entries = []
    support = []
    for index, fraction in enumerate(weights):
        if float(fraction) > cutoff:
            support.append(wavevectors[index])
            entries.append(
                {
                    "index": int(index),
                    "wavevector": [int(component) for component in wavevectors[index]],
                    "shell": int(k2s[index]),
                    "x_fraction": float(fraction),
                }
            )
    return frozenset(support), entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strata-json", required=True)
    parser.add_argument("--cutoff", type=float, default=1e-6)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = json.loads(Path(args.strata_json).read_text(encoding="utf-8"))
    orbit = active_orbit_supports()
    dca = {8, 10, 14}
    rows = []
    for row_index, row in enumerate(data["rows"]):
        if not dca.issubset(set(int(shell) for shell in row["shells"])):
            continue
        support, entries = significant_support(row, args.cutoff)
        rows.append(
            {
                "row_index": row_index,
                "shells": [int(shell) for shell in row["shells"]],
                "best_value": float(row["best_value"]),
                "gap_to_target": float(data["target"] - row["best_value"]),
                "support_size": len(support),
                "orbit_match": support in orbit,
                "support": entries,
            }
        )
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": args.strata_json,
        "target": data["target"],
        "cutoff": args.cutoff,
        "active_orbit_support_count": len(orbit),
        "dca_rows": len(rows),
        "all_dca_rows_orbit_match": bool(rows and all(row["orbit_match"] for row in rows)),
        "all_dca_rows_support_size_9": bool(rows and all(row["support_size"] == 9 for row in rows)),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"DCA rows: {len(rows)}")
    print(f"all orbit match: {report['all_dca_rows_orbit_match']}")
    print(f"all support size 9: {report['all_dca_rows_support_size_9']}")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()