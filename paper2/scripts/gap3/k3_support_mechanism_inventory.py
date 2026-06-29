#!/usr/bin/env python3
"""Canonicalize k=3 support mechanisms under signed coordinate permutations."""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable


def positive_rep(mode: tuple[int, int, int]) -> tuple[int, int, int]:
    negative = tuple(-component for component in mode)
    return mode if mode > negative else negative


def support_images(support: Iterable[tuple[int, int, int]]) -> Iterable[tuple[tuple[int, int, int], ...]]:
    modes = [tuple(int(component) for component in mode) for mode in support]
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            image = []
            for mode in modes:
                mapped = tuple(signs[index] * mode[perm[index]] for index in range(3))
                image.append(positive_rep(mapped))
            yield tuple(sorted(image))


def canonical_support(support: Iterable[tuple[int, int, int]]) -> tuple[tuple[int, int, int], ...]:
    return min(support_images(support))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kkt-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = json.loads(Path(args.kkt_json).read_text(encoding="utf-8"))
    groups: dict[tuple[tuple[int, int, int], ...], list[dict]] = defaultdict(list)
    for row in data["rows"]:
        support = [tuple(int(component) for component in mode) for mode in row["support_modes"]]
        groups[canonical_support(support)].append(row)

    mechanisms = []
    for index, (canonical, rows) in enumerate(sorted(groups.items(), key=lambda item: max(row["support_value"] for row in item[1]), reverse=True)):
        best = max(rows, key=lambda row: row["support_value"])
        mechanisms.append(
            {
                "mechanism_index": index,
                "canonical_support": [list(mode) for mode in canonical],
                "support_size": len(canonical),
                "cluster_indices": sorted(int(row["cluster_index"]) for row in rows),
                "cluster_count": len(rows),
                "best_value": float(best["support_value"]),
                "best_gap_to_target": float(best["support_gap_to_target"]),
                "best_min_release": float(best["full_release"]["minimum"]["min_release_coefficient"]),
                "best_hessian_negative_count": int(best["hessian_minus_R"]["negative"]),
                "best_hessian_flat_count": int(best["hessian_minus_R"]["flat"]),
                "best_hessian_positive_count": int(best["hessian_minus_R"]["positive"]),
            }
        )

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": args.kkt_json,
        "target": data["target"],
        "clusters": len(data["rows"]),
        "mechanisms": len(mechanisms),
        "best_mechanism": mechanisms[0] if mechanisms else None,
        "mechanism_rows": mechanisms,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"clusters: {report['clusters']}")
    print(f"canonical mechanisms: {report['mechanisms']}")
    if mechanisms:
        print(f"best: support={mechanisms[0]['support_size']} value={mechanisms[0]['best_value']:.15f} gap={mechanisms[0]['best_gap_to_target']:+.6e}")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()