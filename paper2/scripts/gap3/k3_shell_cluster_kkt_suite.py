#!/usr/bin/env python3
"""KKT suite for non-DCA shell-stratum clusters in the k=3 block.

This script turns the shell-stratum scan ledger into finite KKT data.  For each
distinct non-DCA value cluster it:

1. extracts the effective X-energy support from a representative optimizer row;
2. repolishes the exact support problem;
3. checks the Hessian of -R on that support;
4. embeds the support into the full 85-mode block and computes one-mode release
   eigenvalue tests for every omitted full-block mode.

A row with H(-R) nonnegative on the support and positive one-mode release
coefficients is a genuine local full-block KKT competitor; a row with a negative
release coefficient is not a full-block KKT point and cannot be a global
maximizer.  This is still a floating/finite-difference certificate scaffold,
but it is the finite support/KKT object needed to replace broad shell scans.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.gap3.k3_active_set_verify import build_problem_scope  # noqa: E402
from scripts.gap3.k3_inactive_release_eigen import compute_b, release_matrix  # noqa: E402
from scripts.gap3.k3_shell_strata_scan import C3_TARGET, objective  # noqa: E402
from scripts.gap3.multi_mode_beta_bound import divfree_basis, precompute_triads  # noqa: E402


def resolve_workers(requested: int, task_count: int) -> int:
    if requested == 0:
        return min(max(1, os.cpu_count() or 1), max(1, task_count))
    return min(max(1, requested), max(1, task_count))


def parse_cluster_indices(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def build_support_problem(wavevectors: list[tuple[int, int, int]]) -> dict[str, Any]:
    _, ell_idx, ell2, r_idx, s_idx, s_mat = precompute_triads(wavevectors)
    return {
        "wavevecs": [tuple(int(component) for component in wavevector) for wavevector in wavevectors],
        "N": len(wavevectors),
        "k2s": np.asarray([sum(component * component for component in wavevector) for wavevector in wavevectors], dtype=float),
        "e1s": np.asarray([divfree_basis(wavevector)[0] for wavevector in wavevectors], dtype=float),
        "e2s": np.asarray([divfree_basis(wavevector)[1] for wavevector in wavevectors], dtype=float),
        "ell_idx": ell_idx,
        "ell2": ell2,
        "r_idx": r_idx,
        "s_idx": s_idx,
        "s_mat": s_mat,
    }


def cluster_non_dca_rows(rows: list[dict[str, Any]], tolerance: float) -> list[dict[str, Any]]:
    dca = {8, 10, 14}
    clusters: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        if dca.issubset(set(int(shell) for shell in row["shells"])):
            continue
        for cluster in clusters:
            if abs(float(row["best_value"]) - float(cluster["value"])) <= tolerance:
                cluster["rows"].append({"row_index": row_index, **row})
                break
        else:
            clusters.append({"value": float(row["best_value"]), "rows": [{"row_index": row_index, **row}]})
    clusters.sort(key=lambda item: item["value"], reverse=True)
    return clusters


def effective_support(row: dict[str, Any], cutoff: float) -> tuple[list[tuple[int, int, int]], np.ndarray, list[dict[str, Any]]]:
    params = np.asarray(row["best_params"], dtype=float)
    wavevectors = [tuple(int(component) for component in wavevector) for wavevector in row["wavevectors"]]
    k2s = np.asarray([sum(component * component for component in wavevector) for wavevector in wavevectors], dtype=float)
    x_weights = 2.0 * k2s * np.exp(params[3::4])
    fractions = x_weights / float(np.sum(x_weights))
    indices = [int(index) for index, fraction in enumerate(fractions) if float(fraction) > cutoff]
    support_modes = [wavevectors[index] for index in indices]
    support_params = np.empty(4 * len(indices), dtype=float)
    rows = []
    for target_index, source_index in enumerate(indices):
        support_params[4 * target_index : 4 * target_index + 4] = params[4 * source_index : 4 * source_index + 4]
        rows.append(
            {
                "source_index": source_index,
                "wavevector": list(wavevectors[source_index]),
                "shell": int(k2s[source_index]),
                "x_fraction": float(fractions[source_index]),
                "loga": float(params[4 * source_index + 3]),
            }
        )
    return support_modes, support_params, rows


def polish_support(problem: dict[str, Any], start: np.ndarray, maxiter: int) -> tuple[float, np.ndarray, float]:
    bounds = [(0.0, math.pi / 2.0), (0.0, 2.0 * math.pi), (0.0, 2.0 * math.pi), (-60.0, 60.0)] * int(problem["N"])

    def fun(params: np.ndarray) -> float:
        return float(objective(problem, params)[0])

    def jac(params: np.ndarray) -> np.ndarray:
        return np.asarray(objective(problem, params)[1], dtype=float)

    result = minimize(fun, start, jac=jac, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter, "gtol": 1e-11, "ftol": 1e-15})
    params = normalize_x2(problem, np.asarray(result.x, dtype=float))
    value, gradient = objective(problem, params)
    return -float(value), params, float(np.max(np.abs(gradient)))


def normalize_x2(problem: dict[str, Any], params: np.ndarray, target_x2: float = 1.0) -> np.ndarray:
    normalized = np.asarray(params, dtype=float).copy()
    x2 = 2.0 * float(np.dot(problem["k2s"], np.exp(normalized[3::4])))
    if x2 <= 0.0:
        raise ValueError("cannot normalize a zero support")
    normalized[3::4] += math.log(target_x2 / x2)
    return normalized


def hessian_counts(problem: dict[str, Any], params: np.ndarray, step: float, flat_tol: float) -> dict[str, Any]:
    dimension = len(params)
    matrix = np.zeros((dimension, dimension), dtype=float)
    for index in range(dimension):
        plus = params.copy()
        minus = params.copy()
        plus[index] += step
        minus[index] -= step
        _, grad_plus = objective(problem, plus)
        _, grad_minus = objective(problem, minus)
        matrix[:, index] = (np.asarray(grad_plus, dtype=float) - np.asarray(grad_minus, dtype=float)) / (2.0 * step)
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(matrix)
    return {
        "dimension": dimension,
        "min_eigenvalue": float(eigenvalues[0]),
        "max_eigenvalue": float(eigenvalues[-1]),
        "negative": int(np.sum(eigenvalues < -flat_tol)),
        "flat": int(np.sum(np.abs(eigenvalues) <= flat_tol)),
        "positive": int(np.sum(eigenvalues > flat_tol)),
        "largest": [float(value) for value in eigenvalues[-12:]],
        "smallest": [float(value) for value in eigenvalues[:12]],
    }


def full_base_raw(full_problem: dict[str, Any], support_modes: list[tuple[int, int, int]], params: np.ndarray) -> tuple[np.ndarray, set[int], float, float, float]:
    mode_to_index = {tuple(int(component) for component in mode): index for index, mode in enumerate(full_problem["wavevecs"])}
    full_n = int(full_problem["N"])
    u_pos = np.zeros((full_n, 3), dtype=np.complex128)
    active_indices: set[int] = set()
    for support_index, mode in enumerate(support_modes):
        full_index = mode_to_index[tuple(mode)]
        active_indices.add(full_index)
        theta, phi, psi, loga = params[4 * support_index : 4 * support_index + 4]
        radius = math.sqrt(math.exp(float(loga)))
        u_pos[full_index, :] = radius * (
            math.cos(float(theta)) * np.exp(1j * float(phi)) * full_problem["e1s"][full_index]
            + math.sin(float(theta)) * np.exp(1j * float(psi)) * full_problem["e2s"][full_index]
        )
    u_raw = np.vstack([u_pos, np.conjugate(u_pos)])
    amplitudes = np.sum(np.abs(u_pos) ** 2, axis=1)
    x2 = 2.0 * float(np.dot(full_problem["k2s"], amplitudes))
    d2 = 2.0 * float(np.dot(full_problem["k2s"] ** 2, amplitudes))
    b0 = compute_b(full_problem, u_raw)
    return u_raw, active_indices, x2, d2, b0


def release_summary(full_problem: dict[str, Any], support_modes: list[tuple[int, int, int]], params: np.ndarray, top: int) -> dict[str, Any]:
    base_raw, active_indices, x2, d2, b0 = full_base_raw(full_problem, support_modes, params)
    d_value = math.sqrt(d2)
    r_value = b0 / (x2 * d_value)
    rows = []
    for mode_index in range(int(full_problem["N"])):
        if mode_index in active_indices:
            continue
        matrix = release_matrix(full_problem, base_raw, b0, mode_index)
        eigenvalues = np.linalg.eigvalsh(matrix)
        max_q = float(eigenvalues[-1])
        shell = float(full_problem["k2s"][mode_index])
        denominator_derivative = 2.0 * shell * d_value + x2 * shell * shell / d_value
        denominator_penalty = r_value * denominator_derivative / (x2 * d_value)
        min_release = denominator_penalty - max_q / (x2 * d_value)
        rows.append(
            {
                "mode_index": int(mode_index),
                "wavevector": [int(component) for component in full_problem["wavevecs"][mode_index]],
                "shell": int(shell),
                "min_release_coefficient": float(min_release),
                "denominator_penalty": float(denominator_penalty),
                "max_q_over_unit_polarization": float(max_q),
                "q_eigenvalues": [float(value) for value in eigenvalues],
            }
        )
    rows.sort(key=lambda item: item["min_release_coefficient"])
    return {
        "full_value_from_support": float(r_value),
        "X2": float(x2),
        "D2": float(d2),
        "B": float(b0),
        "minimum": rows[0],
        "negative_count_below_1e-10": int(sum(1 for row in rows if row["min_release_coefficient"] < -1e-10)),
        "smallest_release_rows": rows[:top],
    }


def run_cluster(task: tuple[int, dict[str, Any], float, int, float, float, int]) -> dict[str, Any]:
    cluster_index, cluster, support_cutoff, maxiter, hessian_step, flat_tol, top_release = task
    representative = max(cluster["rows"], key=lambda row: (len(row.get("wavevectors", [])), row["best_value"]))
    support_modes, support_start, support_rows = effective_support(representative, support_cutoff)
    support_problem = build_support_problem(support_modes)
    support_value, support_params, support_grad = polish_support(support_problem, support_start, maxiter)
    hessian = hessian_counts(support_problem, support_params, hessian_step, flat_tol)
    full_problem = build_problem_scope("full-block")
    release = release_summary(full_problem, support_modes, support_params, top_release)
    return {
        "cluster_index": cluster_index,
        "cluster_value_from_shell_scan": float(cluster["value"]),
        "cluster_row_count": len(cluster["rows"]),
        "representative_source_row": int(representative["row_index"]),
        "representative_shells": [int(shell) for shell in representative["shells"]],
        "support_size": len(support_modes),
        "support_modes": [list(mode) for mode in support_modes],
        "support_rows": support_rows,
        "support_params_normalized_x2_1": [float(value) for value in support_params],
        "support_value": float(support_value),
        "support_gap_to_target": float(C3_TARGET - support_value),
        "support_gradient_max_abs": float(support_grad),
        "hessian_minus_R": hessian,
        "full_release": release,
        "full_kkt_status": "release_negative" if release["negative_count_below_1e-10"] else "release_nonnegative",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strata-json", required=True)
    parser.add_argument("--cluster-tol", type=float, default=1e-9)
    parser.add_argument("--support-cutoff", type=float, default=1e-6)
    parser.add_argument("--clusters", default="", help="comma-separated non-DCA cluster indices; empty means selected by --max-clusters/all")
    parser.add_argument("--max-clusters", type=int, default=0, help="0 means all non-DCA clusters")
    parser.add_argument("--maxiter", type=int, default=3000)
    parser.add_argument("--hessian-step", type=float, default=1e-5)
    parser.add_argument("--flat-tol", type=float, default=1e-7)
    parser.add_argument("--top-release", type=int, default=12)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    start_time = time.time()
    data = json.loads(Path(args.strata_json).read_text(encoding="utf-8"))
    all_clusters = cluster_non_dca_rows(data["rows"], args.cluster_tol)
    if args.clusters.strip():
        selected_indices = parse_cluster_indices(args.clusters)
    elif args.max_clusters:
        selected_indices = list(range(min(args.max_clusters, len(all_clusters))))
    else:
        selected_indices = list(range(len(all_clusters)))
    clusters = [(index, all_clusters[index]) for index in selected_indices]
    workers = resolve_workers(args.workers, len(clusters))
    tasks = [
        (index, cluster, args.support_cutoff, args.maxiter, args.hessian_step, args.flat_tol, args.top_release)
        for index, cluster in clusters
    ]
    print("k=3 non-DCA shell-cluster KKT suite")
    print("====================================")
    print(f"clusters={len(clusters)}/{len(all_clusters)} workers={workers} support_cutoff={args.support_cutoff:g}")
    rows: list[dict[str, Any]] = []
    if workers == 1:
        iterator = (run_cluster(task) for task in tasks)
        for row in iterator:
            rows.append(row)
            print(
                f"cluster {row['cluster_index']:2d}: support={row['support_size']:2d} value={row['support_value']:.15f} "
                f"gap={row['support_gap_to_target']:+.3e} min_release={row['full_release']['minimum']['min_release_coefficient']:+.3e} "
                f"neg_release={row['full_release']['negative_count_below_1e-10']} Hneg={row['hessian_minus_R']['negative']}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_cluster, task) for task in tasks]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                print(
                    f"cluster {row['cluster_index']:2d}: support={row['support_size']:2d} value={row['support_value']:.15f} "
                    f"gap={row['support_gap_to_target']:+.3e} min_release={row['full_release']['minimum']['min_release_coefficient']:+.3e} "
                    f"neg_release={row['full_release']['negative_count_below_1e-10']} Hneg={row['hessian_minus_R']['negative']}",
                    flush=True,
                )
    rows.sort(key=lambda row: row["support_value"], reverse=True)
    negative_release = [row for row in rows if row["full_release"]["negative_count_below_1e-10"]]
    negative_hessian = [row for row in rows if row["hessian_minus_R"]["negative"]]
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "non_dca_shell_cluster_kkt_suite",
        "source_json": args.strata_json,
        "target": C3_TARGET,
        "cluster_tolerance": args.cluster_tol,
        "support_cutoff": args.support_cutoff,
        "workers": workers,
        "non_dca_clusters_total": len(all_clusters),
        "selected_cluster_indices": selected_indices,
        "clusters_processed": len(rows),
        "negative_release_clusters": len(negative_release),
        "release_nonnegative_clusters": len(rows) - len(negative_release),
        "negative_hessian_clusters": len(negative_hessian),
        "best_by_value": rows[:10],
        "rows": rows,
        "elapsed_seconds": time.time() - start_time,
        "method": "cluster non-DCA shell rows, repolish effective supports, support Hessian of -R, one-mode full-block release eigenchecks",
    }
    print("\nSummary")
    print(f"  processed={len(rows)} negative_release={len(negative_release)} release_nonnegative={len(rows)-len(negative_release)} negative_hessian={len(negative_hessian)}")
    if rows:
        print(f"  best value={rows[0]['support_value']:.15f} gap={rows[0]['support_gap_to_target']:+.6e}")
    if not args.no_save:
        output = Path(args.output) if args.output else Path("results") / f"k3_shell_cluster_kkt_suite_{datetime.now():%Y%m%d_%H%M%S}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved: {output}")


if __name__ == "__main__":
    main()