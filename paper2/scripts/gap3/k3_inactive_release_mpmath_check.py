#!/usr/bin/env python3
"""High-precision one-sided inactive-release check for the k=3 candidate.

The float64 inactive-release scan estimates d(-R)/d(a_j) from the analytic
log-amplitude gradient.  At very low release floors the raw gradient is near
double precision noise.  This script evaluates the same coefficient from
mpmath objective differences:

    (R(base) - R(base with inactive amplitude a_j)) / a_j.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import mpmath as mp
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.gap3.certify_block_maximum import _build_mpmath_data, _mpmath_objective  # noqa: E402
from scripts.gap3.k3_active_set_verify import build_problem_scope, embed_support  # noqa: E402
from scripts.gap3.k3_active_hessian_check import refine_active_support  # noqa: E402
from scripts.gap3.k3_closed_form_probe import build_prob, support_from_warm  # noqa: E402


def parse_float_list(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def objective_value(problem_data, params: np.ndarray) -> mp.mpf:
    return _mpmath_objective([mp.mpf(str(value)) for value in params], *problem_data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-json", default="results/k3_inactive_release_scan_20260530_targeted.json")
    parser.add_argument("--floors", default="-20,-24,-28,-32")
    parser.add_argument("--base-floor", type=float, default=-120.0)
    parser.add_argument("--dps", type=int, default=120)
    parser.add_argument("--active-refine-starts", type=int, default=0)
    parser.add_argument("--active-refine-scales", default="1e-4,1e-3,1e-2,5e-2")
    parser.add_argument("--active-refine-seed", type=int, default=20260530)
    parser.add_argument("--scope", choices=("nucleus", "full-block"), default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    mp.mp.dps = args.dps
    scan = json.loads(Path(args.scan_json).read_text(encoding="utf-8"))
    scope = args.scope or scan.get("scope", "nucleus")
    minimum = scan["minimum"]
    mode_index = int(minimum["mode_index"])
    angles = np.array(minimum["best_angles"], dtype=float)

    support_indices, support_modes, support_start = support_from_warm()
    support_problem = build_prob(support_modes)
    support_value, support_params, support_grad, refine_rows = refine_active_support(
        support_problem,
        support_start,
        args.active_refine_starts,
        parse_float_list(args.active_refine_scales),
        args.active_refine_seed,
    )
    full_problem = build_problem_scope(scope)
    base_params, _ = embed_support(full_problem, support_modes, support_params, args.base_floor)
    data = _build_mpmath_data(full_problem, args.dps)
    base_value = objective_value(data, base_params)

    rows = []
    for floor in parse_float_list(args.floors):
        trial = base_params.copy()
        offset = 4 * mode_index
        trial[offset : offset + 3] = angles
        trial[offset + 3] = floor
        trial_value = objective_value(data, trial)
        amplitude = mp.e ** mp.mpf(str(floor))
        coefficient = (base_value - trial_value) / amplitude
        rows.append(
            {
                "floor": floor,
                "amplitude": mp.nstr(amplitude, 40),
                "trial_value": mp.nstr(trial_value, 80),
                "coefficient": mp.nstr(coefficient, 80),
            }
        )
        print(f"floor={floor:6g} coefficient={mp.nstr(coefficient, 50)}")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scan_json": args.scan_json,
        "dps": args.dps,
        "base_floor": args.base_floor,
        "base_value": mp.nstr(base_value, 80),
        "support_value_float64": support_value,
        "support_grad_max": support_grad,
        "active_refine_starts_per_scale": args.active_refine_starts,
        "active_refine_top": sorted(refine_rows, key=lambda item: item["value"], reverse=True)[:20],
        "scope": scope,
        "problem_modes": int(full_problem["N"]),
        "problem_triads": int(len(full_problem["ell_idx"])),
        "mode_index": mode_index,
        "wavevector": minimum["wavevector"],
        "angles": [float(value) for value in angles],
        "coefficients": rows,
    }

    if not args.no_save:
        output = Path(args.output) if args.output else Path("results") / f"k3_inactive_release_mpmath_check_{datetime.now():%Y%m%d_%H%M%S}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved: {output}")


if __name__ == "__main__":
    main()