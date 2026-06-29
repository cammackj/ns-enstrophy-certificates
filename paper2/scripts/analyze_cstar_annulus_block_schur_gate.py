"""Block signed stability diagnostic for the C* annulus branch.

This is the first Gate-D diagnostic.  It does not stream triads.  It reuses the
completed one-high direction caches, evaluates the exact signed numerator by an
alias-safe FFT, fits the multiblock cubic numerator polynomial, and optimizes
independent block scales.

The purpose is to determine whether the common-scale full-1413 branch is even
stationary against rank-window releases before building a finer Schur
certificate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.fft as scipy_fft
from scipy.optimize import minimize, minimize_scalar

from optimize_cstar_release_band import load_base_coeffs
from probe_cstar_annulus_remainder_direction import (
    load_direction_cache_parts,
    modes_on_shells,
    selected_shells_from_optimizer,
)
from probe_cstar_inactive_gradient import DEFAULT_COEFF_DIR, basis_arrays, positive_modes


@dataclass(frozen=True)
class BlockSpec:
    label: str
    rank_start: int
    rank_count: int
    cache_parts: tuple[Path, ...]


def cache_paths_for_range(results_dir: Path, rank_start: int, rank_end: int) -> tuple[Path, ...]:
    def rank_path(lo: int, hi: int) -> Path:
        return results_dir / f"cstar_annulus_direction_rank{lo}_{hi}.npz"

    paths: list[Path] = []
    if rank_start == 1:
        if rank_end < 10:
            raise ValueError("rank ranges starting at 1 require rank_end >= 10")
        paths.append(results_dir / "cstar_annulus_direction_top10.npz")
        first = 11
    else:
        first = rank_start
    if (first - 1) % 10 != 0 or rank_end % 10 != 0:
        raise ValueError(f"rank range must align to cached tens after top10: {rank_start}..{rank_end}")
    paths.extend(rank_path(lo, lo + 9) for lo in range(first, rank_end + 1, 10))
    return tuple(paths)


def cache_parts_from_group_row(row: dict[str, object], results_dir: Path) -> tuple[Path, ...]:
    if "cache_parts" in row:
        return tuple(Path(str(path)) for path in row["cache_parts"])
    rank_start = int(row["rank_start"])
    rank_end = int(row["rank_end"])
    if rank_start == 701 and rank_end == 1413:
        return (results_dir / "cstar_annulus_direction_rank701_1413.npz",)
    return cache_paths_for_range(results_dir, rank_start, rank_end)


def singleton_split_cache(split_dir: Path, rank: int) -> Path:
    """Return the unique cached one-rank shell split direction for a rank."""
    patterns = [
        f"cstar_annulus_direction_rank{rank:04d}_shell*.npz",
        f"cstar_annulus_direction_rank{rank}_shell*.npz",
    ]
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(split_dir.glob(pattern)))
    unique_matches = sorted(set(matches))
    if len(unique_matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one split cache for rank {rank} in {split_dir}, found {len(unique_matches)}"
        )
    return unique_matches[0]


def residual_split_cache(split_dir: Path, rank_start: int, rank_end: int, explicit: Path | None = None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"explicit residual cache does not exist: {explicit}")
        return explicit
    candidates = [
        split_dir / f"cstar_annulus_direction_rank{rank_start}_{rank_end}_residual.npz",
        split_dir / f"cstar_annulus_direction_rank{rank_start:04d}_{rank_end:04d}_residual.npz",
        split_dir / f"cstar_annulus_direction_rank{rank_start:03d}_{rank_end:03d}_residual.npz",
    ]
    existing = sorted({path for path in candidates if path.exists()})
    if len(existing) != 1:
        raise FileNotFoundError(
            f"expected exactly one residual cache for ranks {rank_start}..{rank_end} in {split_dir}, "
            f"found {len(existing)}"
        )
    return existing[0]


def group_row_from_cache(
    *,
    label: str,
    rank_start: int,
    rank_end: int,
    cache_parts: tuple[Path, ...],
    initial_scale: float,
    csv_path: Path,
    base_s2: int,
) -> dict[str, object]:
    rank_count = rank_end - rank_start + 1
    shells, _, _, _, _ = selected_shells_from_optimizer(csv_path, base_s2, rank_start, rank_count)
    group_pos, _ = modes_on_shells(set(shells), max(shells))
    _, _, _, linear, delta_x, delta_d, _ = load_direction_cache_parts(cache_parts, shells, group_pos)
    return {
        "label": label,
        "rank_start": rank_start,
        "rank_end": rank_end,
        "rank_count": rank_count,
        "shell_count": len(shells),
        "linear": float(linear),
        "delta_x": float(delta_x),
        "delta_d": float(delta_d),
        "initial_scale": float(initial_scale),
        "cache_count": len(cache_parts),
        "cache_parts": [str(path) for path in cache_parts],
    }


def expand_parent_residual_json(
    *,
    source_json: Path,
    parent_label: str,
    child_rank_start: int,
    child_rank_end: int,
    split_dir: Path,
    residual_cache_path: Path | None,
    output_json: Path,
    csv_path: Path,
    base_s2: int,
) -> dict[str, object]:
    """Replace a parent residual group by cached singleton shell splits plus a new residual."""
    summary = json.loads(source_json.read_text(encoding="utf-8"))
    if "scales" not in summary or "group_rows" not in summary:
        raise ValueError("--direct-expand-parent-json requires a JSON with scales/group_rows")
    scales = list(summary["scales"])
    group_rows = [dict(row) for row in summary["group_rows"]]
    if len(scales) != len(group_rows):
        raise ValueError("expand source scales/group_rows length mismatch")

    parent_matches = [index for index, row in enumerate(group_rows) if str(row["label"]) == parent_label]
    if len(parent_matches) != 1:
        raise ValueError(f"expected exactly one parent label {parent_label!r}, found {len(parent_matches)}")
    parent_index = parent_matches[0]
    parent_row = group_rows[parent_index]
    parent_rank_start = int(parent_row["rank_start"])
    parent_rank_end = int(parent_row["rank_end"])
    parent_scale = float(scales[parent_index])
    if child_rank_start < parent_rank_start or child_rank_end >= parent_rank_end:
        raise ValueError(
            f"child range {child_rank_start}..{child_rank_end} must be a strict initial subrange of "
            f"parent {parent_rank_start}..{parent_rank_end}"
        )
    if child_rank_start > child_rank_end:
        raise ValueError("child rank start must be <= child rank end")
    if not split_dir.exists():
        raise FileNotFoundError(f"split directory does not exist: {split_dir}")

    expanded_rows: list[dict[str, object]] = []
    expanded_scales: list[float] = []
    expanded_rows.extend(group_rows[:parent_index])
    expanded_scales.extend(float(value) for value in scales[:parent_index])

    for rank in range(child_rank_start, child_rank_end + 1):
        cache_path = singleton_split_cache(split_dir, rank)
        shells, _, _, _, _ = selected_shells_from_optimizer(csv_path, base_s2, rank, 1)
        if len(shells) != 1:
            raise ValueError(f"rank {rank} singleton split produced {len(shells)} shells")
        expanded_rows.append(
            group_row_from_cache(
                label=f"r{rank:04d}_s{shells[0]}",
                rank_start=rank,
                rank_end=rank,
                cache_parts=(cache_path,),
                initial_scale=parent_scale,
                csv_path=csv_path,
                base_s2=base_s2,
            )
        )
        expanded_scales.append(parent_scale)

    residual_rank_start = child_rank_end + 1
    if residual_rank_start <= parent_rank_end:
        cache_path = residual_split_cache(split_dir, residual_rank_start, parent_rank_end, residual_cache_path)
        expanded_rows.append(
            group_row_from_cache(
                label=f"r{residual_rank_start:03d}_{parent_rank_end:03d}",
                rank_start=residual_rank_start,
                rank_end=parent_rank_end,
                cache_parts=(cache_path,),
                initial_scale=parent_scale,
                csv_path=csv_path,
                base_s2=base_s2,
            )
        )
        expanded_scales.append(parent_scale)

    expanded_rows.extend(group_rows[parent_index + 1 :])
    expanded_scales.extend(float(value) for value in scales[parent_index + 1 :])

    output = dict(summary)
    output.update(
        {
            "expanded_source_json": str(source_json),
            "expanded_parent_label": parent_label,
            "expanded_parent_rank_start": parent_rank_start,
            "expanded_parent_rank_end": parent_rank_end,
            "expanded_parent_scale": parent_scale,
            "expanded_child_rank_start": child_rank_start,
            "expanded_child_rank_end": child_rank_end,
            "expanded_split_dir": str(split_dir),
            "expansion_note": (
                "Audit source produced by replacing one parent residual group with cached singleton "
                "shell-split directions and the remaining cached residual block. Optimization diagnostics "
                "from the source JSON are inherited only as provenance; rerun direct audit/optimization "
                "for gradients in this expanded coordinate system."
            ),
            "group_count": len(expanded_rows),
            "scales": [float(value) for value in expanded_scales],
            "initial_scales": [float(value) for value in expanded_scales],
            "group_rows": expanded_rows,
        }
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output


def windowed_blocks(results_dir: Path, rank_window: int) -> list[BlockSpec]:
    if rank_window <= 0 or rank_window % 10 != 0:
        raise ValueError("--rank-window must be a positive multiple of 10")
    if rank_window > 700:
        raise ValueError("--rank-window cannot exceed 700 for the top-700 cached range")

    blocks: list[BlockSpec] = []
    lo = 1
    while lo <= 700:
        hi = min(lo + rank_window - 1, 700)
        if hi % 10 != 0:
            hi -= hi % 10
        if hi < lo:
            raise ValueError(f"rank-window {rank_window} produced an invalid cached range at {lo}")
        blocks.append(
            BlockSpec(
                f"r{lo:03d}_{hi:03d}",
                lo,
                hi - lo + 1,
                cache_paths_for_range(results_dir, lo, hi),
            )
        )
        lo = hi + 1
    blocks.append(
        BlockSpec("r701_1413", 701, 713, (results_dir / "cstar_annulus_direction_rank701_1413.npz",))
    )
    return blocks


def default_blocks(results_dir: Path, layout: str, rank_window: int | None = None) -> list[BlockSpec]:
    if rank_window is not None:
        return windowed_blocks(results_dir, rank_window)
    if layout == "5":
        top300 = cache_paths_for_range(results_dir, 1, 300)
        return [
            BlockSpec("r001_300", 1, 300, top300),
            BlockSpec("r301_400", 301, 100, cache_paths_for_range(results_dir, 301, 400)),
            BlockSpec("r401_500", 401, 100, cache_paths_for_range(results_dir, 401, 500)),
            BlockSpec("r501_700", 501, 200, cache_paths_for_range(results_dir, 501, 700)),
            BlockSpec("r701_1413", 701, 713, (results_dir / "cstar_annulus_direction_rank701_1413.npz",)),
        ]
    if layout == "8":
        return [
            BlockSpec("r001_100", 1, 100, cache_paths_for_range(results_dir, 1, 100)),
            BlockSpec("r101_200", 101, 100, cache_paths_for_range(results_dir, 101, 200)),
            BlockSpec("r201_300", 201, 100, cache_paths_for_range(results_dir, 201, 300)),
            BlockSpec("r301_400", 301, 100, cache_paths_for_range(results_dir, 301, 400)),
            BlockSpec("r401_500", 401, 100, cache_paths_for_range(results_dir, 401, 500)),
            BlockSpec("r501_600", 501, 100, cache_paths_for_range(results_dir, 501, 600)),
            BlockSpec("r601_700", 601, 100, cache_paths_for_range(results_dir, 601, 700)),
            BlockSpec("r701_1413", 701, 713, (results_dir / "cstar_annulus_direction_rank701_1413.npz",)),
        ]
    if layout == "15":
        blocks: list[BlockSpec] = []
        for lo in range(1, 701, 50):
            hi = lo + 49
            blocks.append(
                BlockSpec(
                    f"r{lo:03d}_{hi:03d}",
                    lo,
                    50,
                    cache_paths_for_range(results_dir, lo, hi),
                )
            )
        blocks.append(
            BlockSpec("r701_1413", 701, 713, (results_dir / "cstar_annulus_direction_rank701_1413.npz",))
        )
        return blocks
    raise ValueError(f"unknown layout {layout!r}")


def parse_direct_audit_markdown(path: Path | None) -> dict[str, object]:
    """Read the small W10 audit markdown table into a machine-checkable dict."""
    if path is None:
        return {}
    text = path.read_text(encoding="utf-8")
    summary: dict[str, object] = {"source": str(path), "line_rows": []}
    patterns = {
        "base_ratio": r"- base ratio: `([^`]+)`",
        "gradient_norm": r"- gradient norm: `([^`]+)`",
        "gradient_max_abs": r"- max absolute gradient component: `([^`]+)`",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            summary[key] = float(match.group(1))

    line_rows: list[dict[str, object]] = []
    for raw_line in text.splitlines():
        if not raw_line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in raw_line.strip().strip("|").split("|")]
        if len(cells) != 7:
            continue
        direction = cells[0].strip("`")
        if direction in {"direction", "---"}:
            continue
        try:
            coordinate_gradient = (
                None if cells[6] in {"", "``"} else float(cells[6].strip("`"))
            )
            line_rows.append(
                {
                    "direction": direction,
                    "step": float(cells[1].strip("`")),
                    "slope": float(cells[2].strip("`")),
                    "curvature": float(cells[3].strip("`")),
                    "plus_ratio": float(cells[4].strip("`")),
                    "minus_ratio": float(cells[5].strip("`")),
                    "coordinate_gradient": coordinate_gradient,
                }
            )
        except ValueError:
            continue
    summary["line_rows"] = line_rows
    gradient_rows = [row for row in line_rows if row["direction"] == "gradient_l2"]
    if gradient_rows:
        summary["gradient_l2_curvature"] = float(gradient_rows[0]["curvature"])
        summary["gradient_l2_step"] = float(gradient_rows[0]["step"])
    return summary


def write_direct_split_proof_budget(
    *,
    source_json: Path,
    audit_md: Path | None,
    hessian_json: Path | None,
    output_json: Path | None,
    output_md: Path | None,
    cpu_ratio: float | None,
    cpu_numerator: float | None,
    cpu_denominator: float | None,
    cpu_imag: float | None,
    replay_epsilon: float,
) -> dict[str, object]:
    summary = json.loads(source_json.read_text(encoding="utf-8"))
    scales = np.asarray(summary["scales"], dtype=np.float64)
    group_rows = list(summary["group_rows"])
    if len(scales) != len(group_rows):
        raise ValueError("proof budget scales/group_rows length mismatch")

    audit = parse_direct_audit_markdown(audit_md)
    hessian_summary = json.loads(hessian_json.read_text(encoding="utf-8")) if hessian_json is not None else None
    gpu_ratio = float(summary["final_ratio"])
    replay_ratio = float(cpu_ratio if cpu_ratio is not None else gpu_ratio)
    replay_numerator = None if cpu_numerator is None else float(cpu_numerator)
    replay_denominator = None if cpu_denominator is None else float(cpu_denominator)
    replay_imag = None if cpu_imag is None else float(cpu_imag)
    gpu_cpu_diff = abs(replay_ratio - gpu_ratio)

    gradient_norm = float(audit.get("gradient_norm", summary["final_gradient_norm"]))
    gradient_max_abs = float(audit.get("gradient_max_abs", summary["final_gradient_max_abs"]))
    curvature = audit.get("gradient_l2_curvature")
    curvature_value = None if curvature is None else float(curvature)
    remaining_gain_estimate = None
    local_budgeted_upper = None
    if curvature_value is not None and curvature_value < 0.0:
        remaining_gain_estimate = gradient_norm * gradient_norm / (2.0 * abs(curvature_value))
        local_budgeted_upper = max(gpu_ratio, replay_ratio) + remaining_gain_estimate + replay_epsilon

    delta_x = np.asarray([float(row["delta_x"]) for row in group_rows], dtype=np.float64)
    delta_d = np.asarray([float(row["delta_d"]) for row in group_rows], dtype=np.float64)
    linear = np.asarray([float(row["linear"]) for row in group_rows], dtype=np.float64)
    scale2 = scales * scales
    x_factor = 1.0 + float(delta_x @ scale2)
    d_factor = 1.0 + float(delta_d @ scale2)
    denominator_from_groups = x_factor * math.sqrt(d_factor)
    numerator_from_ratio = replay_ratio * denominator_from_groups
    tail_row = group_rows[-1]
    tail_label = str(tail_row["label"])
    tail_match = re.fullmatch(r"r(\d+)_(\d+)", tail_label)
    has_residual_row = False
    if tail_match is not None:
        tail_rank_start = int(tail_match.group(1))
        tail_rank_end = int(tail_match.group(2))
        has_residual_row = tail_rank_end - tail_rank_start + 1 > 10
    total_delta_x = float(np.sum(delta_x))
    total_delta_d = float(np.sum(delta_d))
    total_linear = float(np.sum(linear))

    active_hessian = None
    active_quadratic_gain = None
    active_newton_step_norm = None
    active_newton_step_max_abs = None
    hessian_spectral_half_margin = None
    hessian_entry_radius_half_margin = None
    hessian_entry_radius_full_margin = None
    log_hessian_min_eigenvalue = None
    log_hessian_max_eigenvalue = None
    log_hessian_spectral_half_margin = None
    log_hessian_entry_radius_half_margin = None
    log_hessian_entry_radius_full_margin = None
    log_weak_eigenvector_top: list[dict[str, object]] | None = None
    normalized_hessian_min_eigenvalue = None
    normalized_hessian_max_eigenvalue = None
    normalized_hessian_entry_radius_half_margin = None
    normalized_gershgorin_min_margin = None
    normalized_gershgorin_max_margin = None
    normalized_gershgorin_positive_count = None
    normalized_weak_eigenvector_top: list[dict[str, object]] | None = None
    weak_eigenvector_top: list[dict[str, object]] | None = None
    newton_step_top: list[dict[str, object]] | None = None
    if hessian_summary is not None:
        hessian_matrix_payload = hessian_summary.get("full_hessian")
        hessian_min_eigenvalue = hessian_summary.get("full_hessian_min_eigenvalue")
        hessian_max_eigenvalue = hessian_summary.get("full_hessian_max_eigenvalue")
        if hessian_matrix_payload is None and hessian_summary.get("selected_hessian") is not None:
            hessian_matrix_payload = hessian_summary.get("selected_hessian")
            hessian_min_eigenvalue = hessian_summary.get("selected_hessian_min_eigenvalue")
            hessian_max_eigenvalue = hessian_summary.get("selected_hessian_max_eigenvalue")
        if hessian_max_eigenvalue is not None and float(hessian_max_eigenvalue) < 0.0:
            hessian_margin = -float(hessian_max_eigenvalue)
            hessian_dimension = len(hessian_matrix_payload) if hessian_matrix_payload is not None else len(scales)
            hessian_spectral_half_margin = 0.5 * hessian_margin
            hessian_entry_radius_half_margin = hessian_margin / (2.0 * hessian_dimension)
            hessian_entry_radius_full_margin = hessian_margin / float(hessian_dimension)
        if hessian_matrix_payload is not None:
            hessian_matrix = np.asarray(hessian_matrix_payload, dtype=np.float64)
            hessian_matrix = 0.5 * (hessian_matrix + hessian_matrix.T)
            labels = [str(row["label"]) for row in hessian_summary["column_rows"]]
            hessian_gradient = np.asarray(
                [float(row["gradient"]) for row in hessian_summary["column_rows"]],
                dtype=np.float64,
            )
            selected_coordinates = [int(index) for index in hessian_summary.get("selected_coordinates", range(len(labels)))]
            selected_scales = scales[np.asarray(selected_coordinates, dtype=np.int64)]
            eig_values, eig_vectors = np.linalg.eigh(hessian_matrix)
            weak_vector = eig_vectors[:, -1]
            weak_eigenvector_top = [
                {
                    "label": labels[int(index)],
                    "component": float(weak_vector[int(index)]),
                }
                for index in np.argsort(np.abs(weak_vector))[::-1][:12]
            ]
            positive_curvature = -hessian_matrix
            curvature_eigs = np.linalg.eigvalsh(positive_curvature)
            if curvature_eigs[0] > 0.0:
                newton_step = np.linalg.solve(positive_curvature, hessian_gradient)
                active_quadratic_gain = 0.5 * float(hessian_gradient @ newton_step)
                active_newton_step_norm = float(np.linalg.norm(newton_step))
                active_newton_step_max_abs = float(np.max(np.abs(newton_step)))
                newton_step_top = [
                    {
                        "label": labels[int(index)],
                        "step": float(newton_step[int(index)]),
                        "gradient": float(hessian_gradient[int(index)]),
                    }
                    for index in np.argsort(np.abs(newton_step))[::-1][:12]
                ]
                residual_gain = max(
                    remaining_gain_estimate if remaining_gain_estimate is not None else 0.0,
                    active_quadratic_gain,
                )
                local_budgeted_upper = max(gpu_ratio, replay_ratio) + residual_gain + replay_epsilon
            log_hessian_matrix = (
                selected_scales[:, None] * hessian_matrix * selected_scales[None, :]
                + np.diag(selected_scales * hessian_gradient)
            )
            log_eig_values, log_eig_vectors = np.linalg.eigh(log_hessian_matrix)
            log_hessian_min_eigenvalue = float(log_eig_values[0])
            log_hessian_max_eigenvalue = float(log_eig_values[-1])
            log_weak_vector = log_eig_vectors[:, -1]
            log_weak_eigenvector_top = [
                {
                    "label": labels[int(index)],
                    "component": float(log_weak_vector[int(index)]),
                }
                for index in np.argsort(np.abs(log_weak_vector))[::-1][:12]
            ]
            if log_hessian_max_eigenvalue < 0.0:
                log_hessian_margin = -log_hessian_max_eigenvalue
                log_hessian_spectral_half_margin = 0.5 * log_hessian_margin
                log_hessian_entry_radius_half_margin = log_hessian_margin / (2.0 * len(labels))
                log_hessian_entry_radius_full_margin = log_hessian_margin / float(len(labels))
            diagonal_curvatures = -np.diag(hessian_matrix)
            if np.all(diagonal_curvatures > 0.0):
                diagonal_weights = 1.0 / np.sqrt(diagonal_curvatures)
                normalized_hessian_matrix = (
                    diagonal_weights[:, None] * hessian_matrix * diagonal_weights[None, :]
                )
                normalized_eig_values, normalized_eig_vectors = np.linalg.eigh(normalized_hessian_matrix)
                normalized_hessian_min_eigenvalue = float(normalized_eig_values[0])
                normalized_hessian_max_eigenvalue = float(normalized_eig_values[-1])
                normalized_weak_vector = normalized_eig_vectors[:, -1]
                normalized_weak_eigenvector_top = [
                    {
                        "label": labels[int(index)],
                        "component": float(normalized_weak_vector[int(index)]),
                    }
                    for index in np.argsort(np.abs(normalized_weak_vector))[::-1][:12]
                ]
                if normalized_hessian_max_eigenvalue < 0.0:
                    normalized_margin = -normalized_hessian_max_eigenvalue
                    normalized_hessian_entry_radius_half_margin = normalized_margin / (2.0 * len(labels))
                normalized_margins = []
                for row_index in range(normalized_hessian_matrix.shape[0]):
                    row_abs_sum = float(np.sum(np.abs(normalized_hessian_matrix[row_index, :])))
                    normalized_margins.append(
                        float(-normalized_hessian_matrix[row_index, row_index] - row_abs_sum + abs(normalized_hessian_matrix[row_index, row_index]))
                    )
                normalized_gershgorin_min_margin = float(min(normalized_margins))
                normalized_gershgorin_max_margin = float(max(normalized_margins))
                normalized_gershgorin_positive_count = int(sum(value > 0.0 for value in normalized_margins))
        active_hessian = {
            "source": str(hessian_json),
            "is_full_hessian": bool(hessian_summary.get("is_full_hessian")),
            "active_only": bool(hessian_summary.get("active_only")),
            "step": float(hessian_summary["step"]),
            "elapsed_seconds": float(hessian_summary["elapsed_seconds"]),
            "gradient_norm": float(hessian_summary["gradient_norm"]),
            "gradient_max_abs": float(hessian_summary["gradient_max_abs"]),
            "full_hessian_min_eigenvalue": hessian_min_eigenvalue,
            "full_hessian_max_eigenvalue": hessian_max_eigenvalue,
            "full_hessian_symmetry_max_abs": hessian_summary.get("full_hessian_symmetry_max_abs"),
            "gradient_direction_curvature": hessian_summary.get("gradient_direction_curvature"),
            "active_quadratic_gain": active_quadratic_gain,
            "active_newton_step_norm": active_newton_step_norm,
            "active_newton_step_max_abs": active_newton_step_max_abs,
            "hessian_spectral_half_margin": hessian_spectral_half_margin,
            "hessian_entry_radius_half_margin": hessian_entry_radius_half_margin,
            "hessian_entry_radius_full_margin": hessian_entry_radius_full_margin,
            "log_hessian_min_eigenvalue": log_hessian_min_eigenvalue,
            "log_hessian_max_eigenvalue": log_hessian_max_eigenvalue,
            "log_hessian_spectral_half_margin": log_hessian_spectral_half_margin,
            "log_hessian_entry_radius_half_margin": log_hessian_entry_radius_half_margin,
            "log_hessian_entry_radius_full_margin": log_hessian_entry_radius_full_margin,
            "normalized_hessian_min_eigenvalue": normalized_hessian_min_eigenvalue,
            "normalized_hessian_max_eigenvalue": normalized_hessian_max_eigenvalue,
            "normalized_hessian_entry_radius_half_margin": normalized_hessian_entry_radius_half_margin,
            "normalized_gershgorin_min_margin": normalized_gershgorin_min_margin,
            "normalized_gershgorin_max_margin": normalized_gershgorin_max_margin,
            "normalized_gershgorin_positive_count": normalized_gershgorin_positive_count,
            "weak_eigenvector_top": weak_eigenvector_top,
            "log_weak_eigenvector_top": log_weak_eigenvector_top,
            "normalized_weak_eigenvector_top": normalized_weak_eigenvector_top,
            "newton_step_top": newton_step_top,
        }

    active_status = "locally optimized at replay precision"
    active_blocker = "interval gradient and Hessian/Schur matrix on the active W10 face"
    if active_hessian is not None and active_hessian["is_full_hessian"]:
        active_status = "floating direct Hessian negative on the full W10 scale face"
        active_blocker = "outward-rounded interval Hessian/Schur matrix on the active W10 face"
    elif (
        active_hessian is not None
        and active_hessian.get("active_only")
        and active_hessian.get("full_hessian_max_eigenvalue") is not None
        and float(active_hessian["full_hessian_max_eigenvalue"]) < 0.0
    ):
        active_status = "floating direct Hessian negative on the active W10 face"
        active_blocker = "outward-rounded interval Hessian/Schur matrix plus zero-face complementarity"

    proof_budget: dict[str, object] = {
        "source_json": str(source_json),
        "audit_md": None if audit_md is None else str(audit_md),
        "hessian_json": None if hessian_json is None else str(hessian_json),
        "candidate": {
            "group_count": len(group_rows),
            "layout": summary.get("layout"),
            "fft_grid": int(summary["fft_grid"]),
            "alias_safe_coordinate_bound": 47,
            "alias_safe_requirement": "G > 3K = 141",
            "backend": summary.get("backend"),
            "torch_precision": summary.get("torch_precision"),
            "gpu_ratio": gpu_ratio,
            "cpu_replay_ratio": replay_ratio,
            "cpu_replay_numerator": replay_numerator,
            "cpu_replay_denominator": replay_denominator,
            "cpu_replay_imag": replay_imag,
            "gpu_cpu_difference": gpu_cpu_diff,
        },
        "stationarity": {
            "gradient_norm": gradient_norm,
            "gradient_max_abs": gradient_max_abs,
            "gradient_l2_curvature": curvature_value,
            "remaining_gain_estimate": remaining_gain_estimate,
            "active_quadratic_gain": active_quadratic_gain,
            "replay_epsilon": replay_epsilon,
            "local_budgeted_upper": local_budgeted_upper,
        },
        "active_hessian": active_hessian,
        "group_norms": {
            "scale_min": float(np.min(scales)),
            "scale_max": float(np.max(scales)),
            "total_delta_x": total_delta_x,
            "total_delta_d": total_delta_d,
            "total_linear": total_linear,
            "scaled_delta_x": float(delta_x @ scale2),
            "scaled_delta_d": float(delta_d @ scale2),
            "x_factor": x_factor,
            "d_factor": d_factor,
            "denominator_from_groups": denominator_from_groups,
            "numerator_from_ratio": numerator_from_ratio,
            "tail_label": tail_label,
            "has_residual_row": has_residual_row,
            "tail_delta_x_fraction": float(float(tail_row["delta_x"]) / total_delta_x),
            "tail_delta_d_fraction": float(float(tail_row["delta_d"]) / total_delta_d),
            "tail_linear_fraction": float(float(tail_row["linear"]) / total_linear),
        },
        "proof_gates": [
            {
                "gate": "alias-safe finite replay",
                "status": "numerically replayed; interval wrapper still required",
                "blocking_item": "outward-rounded numerator/denominator replay or explicit gamma_n bound",
            },
            {
                "gate": "active W10 stationarity",
                "status": active_status,
                "blocking_item": active_blocker,
            },
            {
                "gate": "inactive complementarity",
                "status": "not transferred from old top-700/common branch",
                "blocking_item": "W10-specific inactive score rows or a certified transfer lemma",
            },
            {
                "gate": "tail transfer",
                "status": "existing far-tail machinery available",
                "blocking_item": "interface the W10 finite certificate with the far-tail theorem",
            },
        ],
    }

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(proof_budget, indent=2) + "\n", encoding="utf-8")

    if output_md is not None:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        stationarity = proof_budget["stationarity"]
        candidate = proof_budget["candidate"]
        group_norms = proof_budget["group_norms"]
        local_upper = stationarity["local_budgeted_upper"]
        lines = [
            "# C* W10/Adaptive Proof Budget",
            "",
            "This is a tracking artifact, not manuscript text.  It records the current",
            "W10/adaptive branch as a theorem-gate object and separates the replayed",
            "candidate value from the remaining interval Schur/KKT work.",
            "",
            "## Verdict",
            "",
            "The W10/adaptive branch is an excellent lower-bound and local-stationarity",
            "candidate.  It is not yet a paper-ready equality/global-upper theorem.",
            "The remaining work is specific: interval active Schur/KKT, W10-specific",
            "inactive complementarity, and finite-to-tail transfer.",
            "",
            "## Candidate Replay",
            "",
            f"- source JSON: `{source_json}`",
            f"- audit MD: `{audit_md}`",
            f"- Hessian JSON: `{hessian_json}`",
            f"- layout: `{candidate['layout']}`",
            f"- group count: `{candidate['group_count']}`",
            f"- FFT grid: `{candidate['fft_grid']}`",
            f"- alias safety: `{candidate['alias_safe_requirement']}`",
            f"- GPU ratio: `{gpu_ratio:.17g}`",
            f"- CPU replay ratio: `{replay_ratio:.17g}`",
            f"- CPU/GPU difference: `{gpu_cpu_diff:.3e}`",
        ]
        if replay_numerator is not None:
            lines.append(f"- CPU replay numerator: `{replay_numerator:.17g}`")
        if replay_denominator is not None:
            lines.append(f"- CPU replay denominator: `{replay_denominator:.17g}`")
        if replay_imag is not None:
            lines.append(f"- CPU replay imaginary residue: `{replay_imag:.3e}`")
        lines.extend(
            [
                "",
                "## Stationarity Budget",
                "",
                f"- gradient norm: `{gradient_norm:.17g}`",
                f"- max absolute gradient component: `{gradient_max_abs:.17g}`",
                f"- gradient-direction curvature: `{curvature_value:.17g}`"
                if curvature_value is not None
                else "- gradient-direction curvature: `not recorded`",
                f"- estimated remaining gain along gradient direction: `{remaining_gain_estimate:.17g}`"
                if remaining_gain_estimate is not None
                else "- estimated remaining gain along gradient direction: `not available`",
                f"- full active quadratic residual gain: `{active_quadratic_gain:.17g}`"
                if active_quadratic_gain is not None
                else "- full active quadratic residual gain: `not available`",
                f"- replay safety epsilon used in this ledger: `{replay_epsilon:.17g}`",
                f"- local replay-plus-gradient budget: `{local_upper:.17g}`"
                if local_upper is not None
                else "- local replay-plus-gradient budget: `not available`",
                "",
                "This local budget is not the global theorem.  It only says that, inside",
                "the cached W10 scale variables, the Hessian-resolved active residual",
                "gain is below the displayed budget.  The paper-grade upper theorem",
                "still needs the interval Hessian/Schur certificate.",
                "",
            ]
        )
        if active_hessian is not None:
            lines.extend(
                [
                    "## Active Hessian",
                    "",
                    f"- source: `{active_hessian['source']}`",
                    f"- full Hessian: `{active_hessian['is_full_hessian']}`",
                    f"- active-only Hessian: `{active_hessian['active_only']}`",
                    f"- finite-difference step: `{active_hessian['step']:.17g}`",
                    f"- elapsed seconds: `{active_hessian['elapsed_seconds']:.6g}`",
                    f"- minimum eigenvalue: `{float(active_hessian['full_hessian_min_eigenvalue']):.17g}`"
                    if active_hessian["full_hessian_min_eigenvalue"] is not None
                    else "- minimum eigenvalue: `not available`",
                    f"- maximum eigenvalue: `{float(active_hessian['full_hessian_max_eigenvalue']):.17g}`"
                    if active_hessian["full_hessian_max_eigenvalue"] is not None
                    else "- maximum eigenvalue: `not available`",
                    f"- symmetry max abs: `{float(active_hessian['full_hessian_symmetry_max_abs']):.3e}`"
                    if active_hessian["full_hessian_symmetry_max_abs"] is not None
                    else "- symmetry max abs: `not available`",
                    f"- gradient-direction curvature from Hessian: `{float(active_hessian['gradient_direction_curvature']):.17g}`"
                    if active_hessian["gradient_direction_curvature"] is not None
                    else "- gradient-direction curvature from Hessian: `not available`",
                    f"- full active quadratic residual gain: `{float(active_hessian['active_quadratic_gain']):.17g}`"
                    if active_hessian["active_quadratic_gain"] is not None
                    else "- full active quadratic residual gain: `not available`",
                    f"- Newton step norm: `{float(active_hessian['active_newton_step_norm']):.17g}`"
                    if active_hessian["active_newton_step_norm"] is not None
                    else "- Newton step norm: `not available`",
                    f"- Newton step max abs: `{float(active_hessian['active_newton_step_max_abs']):.17g}`"
                    if active_hessian["active_newton_step_max_abs"] is not None
                    else "- Newton step max abs: `not available`",
                    f"- spectral half-margin target: `{float(active_hessian['hessian_spectral_half_margin']):.17g}`"
                    if active_hessian["hessian_spectral_half_margin"] is not None
                    else "- spectral half-margin target: `not available`",
                    f"- entrywise radius for half-margin by `||E||_2 <= n ||E||_max`: `{float(active_hessian['hessian_entry_radius_half_margin']):.17g}`"
                    if active_hessian["hessian_entry_radius_half_margin"] is not None
                    else "- entrywise radius for half-margin: `not available`",
                    f"- entrywise radius for full-margin by `||E||_2 <= n ||E||_max`: `{float(active_hessian['hessian_entry_radius_full_margin']):.17g}`"
                    if active_hessian["hessian_entry_radius_full_margin"] is not None
                    else "- entrywise radius for full-margin: `not available`",
                    f"- log-scale minimum eigenvalue: `{float(active_hessian['log_hessian_min_eigenvalue']):.17g}`"
                    if active_hessian["log_hessian_min_eigenvalue"] is not None
                    else "- log-scale minimum eigenvalue: `not available`",
                    f"- log-scale maximum eigenvalue: `{float(active_hessian['log_hessian_max_eigenvalue']):.17g}`"
                    if active_hessian["log_hessian_max_eigenvalue"] is not None
                    else "- log-scale maximum eigenvalue: `not available`",
                    f"- log-scale spectral half-margin target: `{float(active_hessian['log_hessian_spectral_half_margin']):.17g}`"
                    if active_hessian["log_hessian_spectral_half_margin"] is not None
                    else "- log-scale spectral half-margin target: `not available`",
                    f"- log-scale entrywise radius for half-margin: `{float(active_hessian['log_hessian_entry_radius_half_margin']):.17g}`"
                    if active_hessian["log_hessian_entry_radius_half_margin"] is not None
                    else "- log-scale entrywise radius for half-margin: `not available`",
                    f"- curvature-normalized minimum eigenvalue: `{float(active_hessian['normalized_hessian_min_eigenvalue']):.17g}`"
                    if active_hessian["normalized_hessian_min_eigenvalue"] is not None
                    else "- curvature-normalized minimum eigenvalue: `not available`",
                    f"- curvature-normalized maximum eigenvalue: `{float(active_hessian['normalized_hessian_max_eigenvalue']):.17g}`"
                    if active_hessian["normalized_hessian_max_eigenvalue"] is not None
                    else "- curvature-normalized maximum eigenvalue: `not available`",
                    f"- curvature-normalized entrywise radius for half-margin: `{float(active_hessian['normalized_hessian_entry_radius_half_margin']):.17g}`"
                    if active_hessian["normalized_hessian_entry_radius_half_margin"] is not None
                    else "- curvature-normalized entrywise radius for half-margin: `not available`",
                    f"- curvature-normalized Gershgorin min margin: `{float(active_hessian['normalized_gershgorin_min_margin']):.17g}`"
                    if active_hessian["normalized_gershgorin_min_margin"] is not None
                    else "- curvature-normalized Gershgorin min margin: `not available`",
                    f"- curvature-normalized Gershgorin positive rows: `{active_hessian['normalized_gershgorin_positive_count']}`"
                    if active_hessian["normalized_gershgorin_positive_count"] is not None
                    else "- curvature-normalized Gershgorin positive rows: `not available`",
                    "",
                    "This closes the floating active-face Hessian diagnostic for the cached",
                    "W10 scale variables.  It still needs an interval version before it can",
                    "serve as a theorem proof.",
                    "",
                    "### Weakest Eigenvector Support",
                    "",
                    "| rank | group | component |",
                    "|---:|---|---:|",
                ]
            )
            for rank, row in enumerate(active_hessian["weak_eigenvector_top"] or [], start=1):
                lines.append(f"| `{rank}` | `{row['label']}` | `{float(row['component']):.17g}` |")
            lines.extend(
                [
                    "",
                    "### Log-Scale Weakest Eigenvector Support",
                    "",
                    "| rank | group | component |",
                    "|---:|---|---:|",
                ]
            )
            for rank, row in enumerate(active_hessian["log_weak_eigenvector_top"] or [], start=1):
                lines.append(f"| `{rank}` | `{row['label']}` | `{float(row['component']):.17g}` |")
            lines.extend(
                [
                    "",
                    "### Curvature-Normalized Weakest Eigenvector Support",
                    "",
                    "| rank | group | component |",
                    "|---:|---|---:|",
                ]
            )
            for rank, row in enumerate(active_hessian["normalized_weak_eigenvector_top"] or [], start=1):
                lines.append(f"| `{rank}` | `{row['label']}` | `{float(row['component']):.17g}` |")
            lines.extend(
                [
                    "",
                    "### Newton Residual Support",
                    "",
                    "| rank | group | Newton step | gradient |",
                    "|---:|---|---:|---:|",
                ]
            )
            for rank, row in enumerate(active_hessian["newton_step_top"] or [], start=1):
                lines.append(
                    f"| `{rank}` | `{row['label']}` | `{float(row['step']):.17g}` "
                    f"| `{float(row['gradient']):.17g}` |"
                )
            lines.extend(
                [
                    "",
                ]
            )
        lines.extend(
            [
                "## Norm Ledger",
                "",
                f"- scale range: `[{group_norms['scale_min']:.17g}, {group_norms['scale_max']:.17g}]`",
                f"- total delta X: `{group_norms['total_delta_x']:.17g}`",
                f"- total delta D: `{group_norms['total_delta_d']:.17g}`",
                f"- scaled delta X: `{group_norms['scaled_delta_x']:.17g}`",
                f"- scaled delta D: `{group_norms['scaled_delta_d']:.17g}`",
                f"- denominator from group norms: `{group_norms['denominator_from_groups']:.17g}`",
                f"- numerator implied by replay ratio: `{group_norms['numerator_from_ratio']:.17g}`",
                f"- residual row present: `{group_norms['has_residual_row']}`",
                f"- {'residual group' if group_norms['has_residual_row'] else 'final finite group'}: `{group_norms['tail_label']}`",
                f"- {'residual' if group_norms['has_residual_row'] else 'final-group'} delta-X fraction: `{group_norms['tail_delta_x_fraction']:.17g}`",
                f"- {'residual' if group_norms['has_residual_row'] else 'final-group'} delta-D fraction: `{group_norms['tail_delta_d_fraction']:.17g}`",
                f"- {'residual' if group_norms['has_residual_row'] else 'final-group'} linear fraction: `{group_norms['tail_linear_fraction']:.17g}`",
                "",
                "## Proof Gates",
                "",
                "| gate | status | blocking item |",
                "|---|---|---|",
            ]
        )
        for gate in proof_budget["proof_gates"]:
            lines.append(f"| {gate['gate']} | {gate['status']} | {gate['blocking_item']} |")
        lines.extend(
            [
                "",
                "## Next Action",
                "",
                "The next useful implementation is a W10 active Schur/KKT audit that",
                "produces interval bounds for the active gradient and the signed Schur",
                "matrix.  The existing score-band ledger should be used only as a",
                "template until W10-specific inactive rows or a transfer lemma are",
                "actually certified.",
                "",
            ]
        )
        output_md.write_text("\n".join(lines), encoding="utf-8")

    return proof_budget


def monomial_exponents(dim: int, degree: int = 3) -> list[tuple[int, ...]]:
    exponents: list[tuple[int, ...]] = []

    def append_fixed_total(position: int, remaining: int, current: list[int]) -> None:
        if position == dim - 1:
            current.append(remaining)
            exponents.append(tuple(current))
            current.pop()
            return
        for power in range(remaining + 1):
            current.append(power)
            append_fixed_total(position + 1, remaining - power, current)
            current.pop()

    for total in range(degree + 1):
        append_fixed_total(0, total, [])
    return exponents


def monomial_row(z: np.ndarray, exponents: list[tuple[int, ...]]) -> np.ndarray:
    row = np.empty(len(exponents), dtype=np.float64)
    for index, powers in enumerate(exponents):
        value = 1.0
        for coordinate, power in zip(z, powers):
            if power:
                value *= float(coordinate) ** power
        row[index] = value
    return row


def monomial_design(points: np.ndarray, exponents: list[tuple[int, ...]]) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    design = np.empty((points.shape[0], len(exponents)), dtype=np.float64)
    for index, powers in enumerate(exponents):
        value = np.ones(points.shape[0], dtype=np.float64)
        for coordinate, power in enumerate(powers):
            if power == 1:
                value *= points[:, coordinate]
            elif power == 2:
                column = points[:, coordinate]
                value *= column * column
            elif power == 3:
                column = points[:, coordinate]
                value *= column * column * column
            elif power:
                value *= points[:, coordinate] ** power
        design[:, index] = value
    return design


def polynomial_value(z: np.ndarray, coeffs: np.ndarray, exponents: list[tuple[int, ...]]) -> float:
    return float(monomial_row(z, exponents) @ coeffs)


def monomial_value_gradient_hessian(z: np.ndarray, powers: tuple[int, ...]) -> tuple[float, np.ndarray, np.ndarray]:
    dim = len(z)
    active = [(index, power) for index, power in enumerate(powers) if power]
    value = 1.0
    gradient = np.zeros(dim, dtype=np.float64)
    hessian = np.zeros((dim, dim), dtype=np.float64)

    for index, power in active:
        value *= float(z[index]) ** power

    for i, power_i in active:
        derivative = float(power_i)
        for k, power_k in active:
            if k == i:
                if power_k > 1:
                    derivative *= float(z[k]) ** (power_k - 1)
            else:
                derivative *= float(z[k]) ** power_k
        gradient[i] = derivative

    for active_i, (i, power_i) in enumerate(active):
        if power_i >= 2:
            second = float(power_i * (power_i - 1))
            for k, power_k in active:
                if k == i:
                    if power_k > 2:
                        second *= float(z[k]) ** (power_k - 2)
                else:
                    second *= float(z[k]) ** power_k
            hessian[i, i] = second
        for j, power_j in active[active_i + 1 :]:
            second = float(power_i * power_j)
            for k, power_k in active:
                if k == i or k == j:
                    if power_k > 1:
                        second *= float(z[k]) ** (power_k - 1)
                else:
                    second *= float(z[k]) ** power_k
            hessian[i, j] = second
            hessian[j, i] = second

    return value, gradient, hessian


def polynomial_value_gradient_hessian(
    z: np.ndarray,
    coeffs: np.ndarray,
    exponents: list[tuple[int, ...]],
) -> tuple[float, np.ndarray, np.ndarray]:
    dim = len(z)
    value = 0.0
    gradient = np.zeros(dim, dtype=np.float64)
    hessian = np.zeros((dim, dim), dtype=np.float64)

    for coefficient, powers in zip(coeffs, exponents):
        if coefficient == 0.0:
            continue
        monomial, monomial_gradient, monomial_hessian = monomial_value_gradient_hessian(z, powers)
        value += float(coefficient) * monomial
        gradient += float(coefficient) * monomial_gradient
        hessian += float(coefficient) * monomial_hessian

    return value, gradient, hessian


def coefficient_grids(
    pos_modes: list[tuple[int, int, int]],
    coeffs_real: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    grid_size: int,
) -> list[np.ndarray]:
    c1 = coeffs_real[:, :2].copy().view(np.complex128).reshape(-1)
    c2 = coeffs_real[:, 2:].copy().view(np.complex128).reshape(-1)
    velocity = c1[:, None] * e1 + c2[:, None] * e2
    coeffs = [np.zeros((grid_size, grid_size, grid_size), dtype=np.complex128) for _ in range(3)]
    for mode, u_value in zip(pos_modes, velocity):
        positive_index = tuple(component % grid_size for component in mode)
        negative_index = tuple((-component) % grid_size for component in mode)
        for component in range(3):
            coeffs[component][positive_index] = u_value[component]
            coeffs[component][negative_index] = np.conj(u_value[component])
    return coeffs


class FftNumerator:
    def __init__(
        self,
        active_grids: list[np.ndarray],
        block_grids: list[list[np.ndarray]],
        x2: float,
        d2: float,
        grid_size: int,
        backend: str,
        workers: int,
    ) -> None:
        self.active_grids = active_grids
        self.block_grids = block_grids
        self.denominator = x2 * math.sqrt(d2)
        self.backend = backend
        self.workers = workers
        freqs = np.fft.fftfreq(grid_size) * grid_size
        kx, ky, kz = np.meshgrid(freqs, freqs, freqs, indexing="ij")
        self.k_grids = (kx, ky, kz)
        self.k2_grid = kx * kx + ky * ky + kz * kz
        self.scale = grid_size**3

    def ifftn(self, value: np.ndarray) -> np.ndarray:
        if self.backend == "scipy":
            return scipy_fft.ifftn(value, workers=self.workers)
        return np.fft.ifftn(value)

    def __call__(self, z: np.ndarray) -> tuple[float, float]:
        coeffs = [component.copy() for component in self.active_grids]
        for scale, block in zip(z, self.block_grids):
            if scale == 0.0:
                continue
            for component in range(3):
                coeffs[component] += float(scale) * block[component]
        u_physical = [self.ifftn(coeffs[component]) * self.scale for component in range(3)]
        physical_b = 0.0 + 0.0j
        for component in range(3):
            advective = 0.0
            for direction_index, k_grid in enumerate(self.k_grids):
                grad = self.ifftn(1j * k_grid * coeffs[component]) * self.scale
                advective = advective + u_physical[direction_index] * grad
            laplacian = self.ifftn(-self.k2_grid * coeffs[component]) * self.scale
            physical_b += np.mean(advective * laplacian)
        return -float(physical_b.real) / self.denominator, float(physical_b.imag) / self.denominator


class TorchCudaFftNumerator:
    def __init__(
        self,
        active_grids: list[np.ndarray],
        block_grids: list[list[np.ndarray]],
        x2: float,
        d2: float,
        grid_size: int,
        precision: str,
    ) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("torch CUDA backend requested, but torch.cuda.is_available() is false")

        self.torch = torch
        self.device = torch.device("cuda")
        self.complex_dtype = torch.complex128 if precision == "float64" else torch.complex64
        self.real_dtype = torch.float64 if precision == "float64" else torch.float32
        self.active_grids = [
            torch.as_tensor(component, dtype=self.complex_dtype, device=self.device) for component in active_grids
        ]
        self.block_grids = [
            [torch.as_tensor(component, dtype=self.complex_dtype, device=self.device) for component in block]
            for block in block_grids
        ]
        self.denominator = x2 * math.sqrt(d2)
        freqs = torch.fft.fftfreq(grid_size, d=1.0 / grid_size, device=self.device, dtype=self.real_dtype)
        kx, ky, kz = torch.meshgrid(freqs, freqs, freqs, indexing="ij")
        self.k_grids = (kx, ky, kz)
        self.k2_grid = kx * kx + ky * ky + kz * kz
        self.scale = grid_size**3

    def __call__(self, z: np.ndarray) -> tuple[float, float]:
        torch = self.torch
        with torch.no_grad():
            z_tensor = torch.as_tensor(z, dtype=self.real_dtype, device=self.device)
            coeffs = [component.clone() for component in self.active_grids]
            for scale, block in zip(z_tensor, self.block_grids):
                if float(scale.item()) == 0.0:
                    continue
                for component in range(3):
                    coeffs[component].add_(block[component], alpha=scale)
            u_physical = [torch.fft.ifftn(coeffs[component]) * self.scale for component in range(3)]
            physical_b = torch.zeros((), dtype=self.complex_dtype, device=self.device)
            for component in range(3):
                advective = torch.zeros_like(u_physical[component])
                for direction_index, k_grid in enumerate(self.k_grids):
                    grad = torch.fft.ifftn((1j * k_grid) * coeffs[component]) * self.scale
                    advective = advective + u_physical[direction_index] * grad
                laplacian = torch.fft.ifftn(-self.k2_grid * coeffs[component]) * self.scale
                physical_b = physical_b + torch.mean(advective * laplacian)
            value = physical_b / self.denominator
            return -float(value.real.item()), float(value.imag.item())


class SingleDirectionFftNumerator:
    def __init__(
        self,
        candidate_grids: list[np.ndarray],
        x2: float,
        d2: float,
        grid_size: int,
        backend: str,
        workers: int,
    ) -> None:
        self.candidate_grids = candidate_grids
        self.denominator = x2 * math.sqrt(d2)
        self.backend = backend
        self.workers = workers
        freqs = np.fft.fftfreq(grid_size) * grid_size
        kx, ky, kz = np.meshgrid(freqs, freqs, freqs, indexing="ij")
        self.k_grids = (kx, ky, kz)
        self.k2_grid = kx * kx + ky * ky + kz * kz
        self.scale = grid_size**3

    def ifftn(self, value: np.ndarray) -> np.ndarray:
        if self.backend == "scipy":
            return scipy_fft.ifftn(value, workers=self.workers)
        return np.fft.ifftn(value)

    def evaluate(self, direction_grids: list[np.ndarray] | None = None, epsilon: float = 0.0) -> tuple[float, float]:
        if direction_grids is None or epsilon == 0.0:
            coeffs = [component.copy() for component in self.candidate_grids]
        else:
            coeffs = [
                self.candidate_grids[component] + float(epsilon) * direction_grids[component]
                for component in range(3)
            ]
        u_physical = [self.ifftn(coeffs[component]) * self.scale for component in range(3)]
        physical_b = 0.0 + 0.0j
        for component in range(3):
            advective = 0.0
            for direction_index, k_grid in enumerate(self.k_grids):
                grad = self.ifftn(1j * k_grid * coeffs[component]) * self.scale
                advective = advective + u_physical[direction_index] * grad
            laplacian = self.ifftn(-self.k2_grid * coeffs[component]) * self.scale
            physical_b += np.mean(advective * laplacian)
        return -float(physical_b.real) / self.denominator, float(physical_b.imag) / self.denominator


class TorchCudaSingleDirectionFftNumerator:
    def __init__(
        self,
        candidate_grids: list[np.ndarray],
        x2: float,
        d2: float,
        grid_size: int,
        precision: str,
    ) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("torch CUDA backend requested, but torch.cuda.is_available() is false")

        self.torch = torch
        self.device = torch.device("cuda")
        self.complex_dtype = torch.complex128 if precision == "float64" else torch.complex64
        self.real_dtype = torch.float64 if precision == "float64" else torch.float32
        self.candidate_grids = [
            torch.as_tensor(component, dtype=self.complex_dtype, device=self.device) for component in candidate_grids
        ]
        self.denominator = x2 * math.sqrt(d2)
        freqs = torch.fft.fftfreq(grid_size, d=1.0 / grid_size, device=self.device, dtype=self.real_dtype)
        kx, ky, kz = torch.meshgrid(freqs, freqs, freqs, indexing="ij")
        self.k_grids = (kx, ky, kz)
        self.k2_grid = kx * kx + ky * ky + kz * kz
        self.scale = grid_size**3

    def _evaluate_tensors(self, coeffs) -> tuple[float, float]:
        torch = self.torch
        with torch.no_grad():
            u_physical = [torch.fft.ifftn(coeffs[component]) * self.scale for component in range(3)]
            physical_b = torch.zeros((), dtype=self.complex_dtype, device=self.device)
            for component in range(3):
                advective = torch.zeros_like(u_physical[component])
                for direction_index, k_grid in enumerate(self.k_grids):
                    grad = torch.fft.ifftn((1j * k_grid) * coeffs[component]) * self.scale
                    advective = advective + u_physical[direction_index] * grad
                laplacian = torch.fft.ifftn(-self.k2_grid * coeffs[component]) * self.scale
                physical_b = physical_b + torch.mean(advective * laplacian)
            value = physical_b / self.denominator
            return -float(value.real.item()), float(value.imag.item())

    def evaluate_base(self) -> tuple[float, float]:
        return self._evaluate_tensors(self.candidate_grids)

    def direction_tensors(self, direction_grids: list[np.ndarray]):
        torch = self.torch
        return [torch.as_tensor(component, dtype=self.complex_dtype, device=self.device) for component in direction_grids]

    def evaluate_direction(self, direction_tensors, epsilon: float) -> tuple[float, float]:
        coeffs = [
            self.candidate_grids[component] + float(epsilon) * direction_tensors[component]
            for component in range(3)
        ]
        return self._evaluate_tensors(coeffs)


class TorchCudaDirectScaleProblem:
    def __init__(
        self,
        all_pos: list[tuple[int, int, int]],
        base_coeffs_real: np.ndarray,
        selected_indices: np.ndarray,
        group_ids: np.ndarray,
        direction_values: np.ndarray,
        e1: np.ndarray,
        e2: np.ndarray,
        delta_x: np.ndarray,
        delta_d: np.ndarray,
        x2: float,
        d2: float,
        grid_size: int,
        precision: str,
    ) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("direct scale optimization requires CUDA")

        self.torch = torch
        self.device = torch.device("cuda")
        self.real_dtype = torch.float64 if precision == "float64" else torch.float32
        self.complex_dtype = torch.complex128 if precision == "float64" else torch.complex64
        self.base_coeffs = torch.as_tensor(base_coeffs_real, dtype=self.real_dtype, device=self.device)
        self.selected_indices = torch.as_tensor(selected_indices, dtype=torch.long, device=self.device)
        self.group_ids = torch.as_tensor(group_ids, dtype=torch.long, device=self.device)
        self.direction_values = torch.as_tensor(direction_values, dtype=self.real_dtype, device=self.device)
        self.e1 = torch.as_tensor(e1, dtype=self.real_dtype, device=self.device)
        self.e2 = torch.as_tensor(e2, dtype=self.real_dtype, device=self.device)
        zero3 = torch.zeros_like(self.e1)
        self.e1_c = torch.complex(self.e1, zero3)
        self.e2_c = torch.complex(self.e2, zero3)
        self.delta_x = torch.as_tensor(delta_x, dtype=self.real_dtype, device=self.device)
        self.delta_d = torch.as_tensor(delta_d, dtype=self.real_dtype, device=self.device)
        self.base_denominator = x2 * math.sqrt(d2)
        self.grid_size = grid_size
        self.grid_count = grid_size**3
        self.scale = grid_size**3

        pos_flat: list[int] = []
        neg_flat: list[int] = []
        for mode in all_pos:
            positive = tuple(component % grid_size for component in mode)
            negative = tuple((-component) % grid_size for component in mode)
            pos_flat.append((positive[0] * grid_size + positive[1]) * grid_size + positive[2])
            neg_flat.append((negative[0] * grid_size + negative[1]) * grid_size + negative[2])
        self.pos_flat = torch.as_tensor(pos_flat, dtype=torch.long, device=self.device)
        self.neg_flat = torch.as_tensor(neg_flat, dtype=torch.long, device=self.device)

        freqs = torch.fft.fftfreq(grid_size, d=1.0 / grid_size, device=self.device, dtype=self.real_dtype)
        kx, ky, kz = torch.meshgrid(freqs, freqs, freqs, indexing="ij")
        self.k_grids = (kx, ky, kz)
        self.k2_grid = kx * kx + ky * ky + kz * kz

    def coefficient_grids(self, scales):
        torch = self.torch
        coeffs = self.base_coeffs.clone()
        updates = self.direction_values * scales[self.group_ids].unsqueeze(1)
        coeffs.index_add_(0, self.selected_indices, updates)
        c1 = torch.view_as_complex(coeffs[:, :2].contiguous())
        c2 = torch.view_as_complex(coeffs[:, 2:].contiguous())
        velocity = c1.unsqueeze(1) * self.e1_c + c2.unsqueeze(1) * self.e2_c

        grids = []
        for component in range(3):
            flat = torch.zeros(self.grid_count, dtype=self.complex_dtype, device=self.device)
            flat = flat.index_copy(0, self.pos_flat, velocity[:, component])
            flat = flat.index_copy(0, self.neg_flat, velocity[:, component].conj())
            grids.append(flat.reshape((self.grid_size, self.grid_size, self.grid_size)))
        return grids

    def ratio(self, scales):
        torch = self.torch
        coeffs = self.coefficient_grids(scales)
        u_physical = [torch.fft.ifftn(coeffs[component]) * self.scale for component in range(3)]
        physical_b = torch.zeros((), dtype=self.complex_dtype, device=self.device)
        for component in range(3):
            advective = torch.zeros_like(u_physical[component])
            for direction_index, k_grid in enumerate(self.k_grids):
                grad = torch.fft.ifftn((1j * k_grid) * coeffs[component]) * self.scale
                advective = advective + u_physical[direction_index] * grad
            laplacian = torch.fft.ifftn(-self.k2_grid * coeffs[component]) * self.scale
            physical_b = physical_b + torch.mean(advective * laplacian)
        numerator = -physical_b.real / self.base_denominator
        scale2 = scales * scales
        denominator = (1.0 + torch.dot(self.delta_x, scale2)) * torch.sqrt(1.0 + torch.dot(self.delta_d, scale2))
        return numerator / denominator

    def value_and_grad(self, scales_np: np.ndarray) -> tuple[float, np.ndarray]:
        torch = self.torch
        scales = torch.tensor(scales_np, dtype=self.real_dtype, device=self.device, requires_grad=True)
        value = self.ratio(scales)
        loss = -value
        loss.backward()
        torch.cuda.synchronize()
        gradient = -scales.grad.detach().cpu().numpy()
        value_float = float(value.detach().cpu())
        del scales, value, loss
        torch.cuda.empty_cache()
        return value_float, gradient

    def value(self, scales_np: np.ndarray) -> float:
        torch = self.torch
        with torch.no_grad():
            scales = torch.as_tensor(scales_np, dtype=self.real_dtype, device=self.device)
            value = self.ratio(scales)
            torch.cuda.synchronize()
            value_float = float(value.detach().cpu())
            del scales, value
            torch.cuda.empty_cache()
            return value_float


def ratio_from_poly(
    z: np.ndarray,
    coeffs: np.ndarray,
    exponents: list[tuple[int, ...]],
    delta_x: np.ndarray,
    delta_d: np.ndarray,
) -> float:
    z2 = z * z
    numerator = polynomial_value(z, coeffs, exponents)
    denominator = (1.0 + float(delta_x @ z2)) * math.sqrt(1.0 + float(delta_d @ z2))
    return numerator / denominator


def ratio_value_gradient_hessian(
    z: np.ndarray,
    coeffs: np.ndarray,
    exponents: list[tuple[int, ...]],
    delta_x: np.ndarray,
    delta_d: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    numerator, numerator_gradient, numerator_hessian = polynomial_value_gradient_hessian(z, coeffs, exponents)
    z2 = z * z
    x_factor = 1.0 + float(delta_x @ z2)
    d_factor = 1.0 + float(delta_d @ z2)
    denominator = x_factor * math.sqrt(d_factor)

    phi_gradient = 2.0 * delta_x * z / x_factor + delta_d * z / d_factor
    phi_hessian = np.diag(2.0 * delta_x / x_factor + delta_d / d_factor)
    phi_hessian -= np.outer(2.0 * delta_x * z, 2.0 * delta_x * z) / (x_factor * x_factor)
    phi_hessian -= 2.0 * np.outer(delta_d * z, delta_d * z) / (d_factor * d_factor)

    inverse_denominator = 1.0 / denominator
    value = numerator * inverse_denominator
    gradient = inverse_denominator * (numerator_gradient - numerator * phi_gradient)
    hessian = inverse_denominator * (
        numerator_hessian
        - np.outer(numerator_gradient, phi_gradient)
        - np.outer(phi_gradient, numerator_gradient)
        + numerator * (np.outer(phi_gradient, phi_gradient) - phi_hessian)
    )
    return float(value), gradient, 0.5 * (hessian + hessian.T)


def scalar_ratio(
    scale: float,
    coeffs: np.ndarray,
    exponents: list[tuple[int, ...]],
    delta_x: np.ndarray,
    delta_d: np.ndarray,
) -> float:
    z = np.full(len(delta_x), scale, dtype=np.float64)
    return ratio_from_poly(z, coeffs, exponents, delta_x, delta_d)


def finite_hessian(function, z: np.ndarray, step: float) -> np.ndarray:
    dim = len(z)
    hessian = np.zeros((dim, dim), dtype=np.float64)
    f0 = function(z)
    for i in range(dim):
        ei = np.zeros(dim, dtype=np.float64)
        ei[i] = step
        hessian[i, i] = (function(z + ei) - 2.0 * f0 + function(z - ei)) / (step * step)
        for j in range(i + 1, dim):
            ej = np.zeros(dim, dtype=np.float64)
            ej[j] = step
            value = (
                function(z + ei + ej)
                - function(z + ei - ej)
                - function(z - ei + ej)
                + function(z - ei - ej)
            ) / (4.0 * step * step)
            hessian[i, j] = value
            hessian[j, i] = value
    return hessian


def build_sample_points(
    dim: int,
    count: int,
    max_scale: float,
    scalar_probe: float,
    seed: int,
    exponents: list[tuple[int, ...]],
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points: list[np.ndarray] = []
    points.append(np.zeros(dim, dtype=np.float64))
    for scale in (0.5, 1.0, scalar_probe, 1.75):
        points.append(np.full(dim, scale, dtype=np.float64))
    for i in range(dim):
        for scale in (0.75, scalar_probe, 1.75):
            point = np.zeros(dim, dtype=np.float64)
            point[i] = scale
            points.append(point)
    while len(points) < count:
        if len(points) % 3 == 0:
            points.append(rng.uniform(0.0, max_scale, size=dim))
        elif len(points) % 3 == 1:
            points.append(rng.uniform(0.25, max_scale, size=dim))
        else:
            mask = rng.random(dim) < 0.55
            point = np.zeros(dim, dtype=np.float64)
            point[mask] = rng.uniform(0.2, max_scale, size=int(np.count_nonzero(mask)))
            points.append(point)
    matrix = np.vstack(points)
    design = monomial_design(matrix, exponents)
    if np.linalg.matrix_rank(design) < len(exponents):
        raise RuntimeError("sample design is rank deficient")
    return matrix


def write_markdown(
    path: Path,
    summary: dict[str, object],
    block_rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_layout = str(summary.get("layout"))
    layout_name = {"5": "five-block", "8": "eight-block", "15": "fifteen-block"}.get(raw_layout)
    if layout_name is None and raw_layout.startswith("w") and raw_layout[1:].isdigit():
        layout_name = f"{int(raw_layout[1:])}-rank-window"
    if layout_name is None:
        layout_name = f"layout-{raw_layout}"
    layout_label = layout_name
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# C* Annulus {layout_label.title()} Schur Gate Diagnostic\n\n")
        handle.write(
            "Generated by `scripts/analyze_cstar_annulus_block_schur_gate.py`. "
            "This is a Gate-D diagnostic, not a manuscript theorem.  It fits the "
            "signed multiblock cubic numerator by alias-safe FFT and optimizes "
            "independent rank-window scales.\n\n"
        )
        handle.write("## Verdict\n\n")
        if bool(summary["scalar_branch_stationary"]):
            handle.write(
                f"The common-scale full-1413 branch is stationary at this {layout_label} "
                "resolution within the validation tolerance.\n\n"
            )
        else:
            handle.write(
                f"The common-scale full-1413 branch is not stationary at this {layout_label} "
                "resolution.  Structural completeness for the scalar branch is therefore "
                "not closed by the existing data; the next candidate is the optimized "
                "block-scaled branch or a finer Schur certificate explaining why this "
                "coarse drift is not admissible.\n\n"
            )
        handle.write("## Numerical Summary\n\n")
        for key in (
            "fft_grid",
            "alias_safe",
            "backend",
            "fft_workers",
            "torch_precision",
            "fit_samples",
            "validation_samples",
            "fit_residual_max",
            "validation_residual_max",
            "scalar_best_scale",
            "scalar_best_ratio",
            "free_best_ratio",
            "free_gain_over_scalar",
            "free_scales",
            "scalar_transverse_gradient_norm",
            "free_hessian_max_eigenvalue",
        ):
            handle.write(f"- `{key}`: `{summary[key]}`\n")
        handle.write("\n## Blocks\n\n")
        handle.write("| block | ranks | shells | linear | delta X | delta D | cache parts |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in block_rows:
            handle.write(
                f"| `{row['label']}` "
                f"| `{row['rank_start']}..{row['rank_end']}` "
                f"| `{row['shell_count']}` "
                f"| `{row['linear']:.17g}` "
                f"| `{row['delta_x']:.17g}` "
                f"| `{row['delta_d']:.17g}` "
                f"| `{row['cache_count']}` |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "A passing result here would not by itself be the full proof, because "
            f"{layout_label} rank windows are coarser than the shell-scale KKT certificate.  "
            "A failing result is still decisive as a diagnostic: the scalar branch "
            "cannot be declared structurally complete until the displayed block "
            "release is either incorporated into the candidate value or excluded "
            "by a sharper admissibility/stability argument.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("scripts/results"))
    parser.add_argument("--base-s2", type=int, default=565)
    parser.add_argument("--base-coeffs", type=Path, default=Path("scripts/results/release_coeffs/release_s565_from550.npy"))
    parser.add_argument("--coeff-dir", type=Path, default=DEFAULT_COEFF_DIR)
    parser.add_argument("--csv-path", type=Path, default=Path("scripts/results/cstar_one_high_annulus_s565.csv"))
    parser.add_argument("--layout", choices=["5", "8", "15"], default="5")
    parser.add_argument(
        "--rank-window",
        type=int,
        default=None,
        help="Override --layout with cached equal rank windows through top 700, e.g. 30 or 50.",
    )
    parser.add_argument("--fft-grid", type=int, default=160)
    parser.add_argument("--fit-samples", type=int, default=96)
    parser.add_argument("--validation-samples", type=int, default=16)
    parser.add_argument("--sample-scale-max", type=float, default=1.9)
    parser.add_argument("--scale-max", type=float, default=3.0)
    parser.add_argument("--backend", choices=["scipy", "numpy", "torch-cuda"], default="scipy")
    parser.add_argument("--fft-workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--torch-precision", choices=["float64", "float32"], default="float64")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--benchmark-evals", type=int, default=3)
    parser.add_argument("--replay-json", type=Path, default=None)
    parser.add_argument(
        "--derivative-audit-json",
        type=Path,
        default=None,
        help="Audit analytic gradient/Hessian from an existing output JSON without rebuilding FFT grids.",
    )
    parser.add_argument("--derivative-audit-md", type=Path, default=None)
    parser.add_argument(
        "--line-audit-json",
        type=Path,
        default=None,
        help="Direct FFT replay of line curvatures along the weakest analytic Hessian directions.",
    )
    parser.add_argument("--line-audit-md", type=Path, default=None)
    parser.add_argument("--line-audit-top-eigs", type=int, default=3)
    parser.add_argument("--line-audit-steps", type=str, default="0.05,0.1,0.2")
    parser.add_argument(
        "--split-audit-json",
        type=Path,
        default=None,
        help="Direct FFT audit of cached rank-window split directions inside the fitted branch.",
    )
    parser.add_argument("--split-audit-md", type=Path, default=None)
    parser.add_argument("--split-audit-window", type=int, default=10)
    parser.add_argument("--split-audit-max-rank", type=int, default=700)
    parser.add_argument("--split-audit-steps", type=str, default="0.05,0.1")
    parser.add_argument(
        "--direct-optimize-json",
        type=Path,
        default=None,
        help="Optimize cached split scales by direct CUDA FFT/autograd, initialized from an existing branch JSON.",
    )
    parser.add_argument("--direct-optimize-md", type=Path, default=None)
    parser.add_argument("--direct-optimize-output-json", type=Path, default=None)
    parser.add_argument("--direct-optimize-window", type=int, default=10)
    parser.add_argument("--direct-optimize-max-rank", type=int, default=700)
    parser.add_argument("--direct-optimize-maxiter", type=int, default=80)
    parser.add_argument("--direct-optimize-gtol", type=float, default=1e-10)
    parser.add_argument("--direct-optimize-objective-scale", type=float, default=1e8)
    parser.add_argument("--direct-optimize-method", choices=["normalized-gradient", "scipy"], default="normalized-gradient")
    parser.add_argument(
        "--direct-optimize-use-source-groups",
        action="store_true",
        help="Optimize exactly the group_rows/cache_parts stored in --direct-optimize-json.",
    )
    parser.add_argument("--direct-optimize-step", type=float, default=0.05)
    parser.add_argument("--direct-optimize-backtracks", type=int, default=12)
    parser.add_argument(
        "--direct-audit-json",
        type=Path,
        default=None,
        help="Audit direct FFT stationarity and line curvatures for a saved split-scale branch.",
    )
    parser.add_argument("--direct-audit-md", type=Path, default=None)
    parser.add_argument("--direct-audit-output-json", type=Path, default=None)
    parser.add_argument("--direct-audit-top-coordinates", type=int, default=8)
    parser.add_argument("--direct-audit-steps", type=str, default="0.01,0.02,0.04")
    parser.add_argument(
        "--direct-hessian-json",
        type=Path,
        default=None,
        help="Finite-difference direct CUDA gradients to audit the active W10 Hessian.",
    )
    parser.add_argument("--direct-hessian-output-json", type=Path, default=None)
    parser.add_argument("--direct-hessian-md", type=Path, default=None)
    parser.add_argument("--direct-hessian-step", type=float, default=0.01)
    parser.add_argument(
        "--direct-hessian-active-only",
        action="store_true",
        help="Compute Hessian columns only for strictly positive scale coordinates.",
    )
    parser.add_argument(
        "--direct-hessian-free-only",
        action="store_true",
        help="Compute Hessian columns only for coordinates strictly inside [0, scale_max].",
    )
    parser.add_argument(
        "--direct-hessian-max-coordinates",
        type=int,
        default=0,
        help="Limit Hessian columns for benchmarking; 0 means all coordinates.",
    )
    parser.add_argument(
        "--proof-budget-json",
        type=Path,
        default=None,
        help="Write a cheap theorem-gate budget from a saved direct split optimization JSON.",
    )
    parser.add_argument("--proof-budget-audit-md", type=Path, default=None)
    parser.add_argument("--proof-budget-hessian-json", type=Path, default=None)
    parser.add_argument("--proof-budget-output-json", type=Path, default=None)
    parser.add_argument("--proof-budget-md", type=Path, default=None)
    parser.add_argument("--proof-budget-cpu-ratio", type=float, default=None)
    parser.add_argument("--proof-budget-cpu-numerator", type=float, default=None)
    parser.add_argument("--proof-budget-cpu-denominator", type=float, default=None)
    parser.add_argument("--proof-budget-cpu-imag", type=float, default=None)
    parser.add_argument("--proof-budget-replay-epsilon", type=float, default=1e-13)
    parser.add_argument(
        "--direct-expand-parent-json",
        type=Path,
        default=None,
        help="Cheaply replace one parent residual group by cached singleton shell splits plus a residual group.",
    )
    parser.add_argument("--direct-expand-parent-label", type=str, default=None)
    parser.add_argument("--direct-expand-rank-start", type=int, default=None)
    parser.add_argument("--direct-expand-rank-end", type=int, default=None)
    parser.add_argument("--direct-expand-split-dir", type=Path, default=None)
    parser.add_argument("--direct-expand-residual-cache", type=Path, default=None)
    parser.add_argument("--direct-expand-output-json", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--hessian-step", type=float, default=1e-4)
    parser.add_argument("--output-md", type=Path, default=Path("references/CSTAR_ANNULUS_BLOCK_SCHUR_GATE_20260604.md"))
    parser.add_argument("--output-json", type=Path, default=Path("scripts/results/cstar_annulus_block_schur_gate_20260604.json"))
    args = parser.parse_args()

    if args.proof_budget_json is not None:
        budget = write_direct_split_proof_budget(
            source_json=args.proof_budget_json,
            audit_md=args.proof_budget_audit_md,
            hessian_json=args.proof_budget_hessian_json,
            output_json=args.proof_budget_output_json,
            output_md=args.proof_budget_md,
            cpu_ratio=args.proof_budget_cpu_ratio,
            cpu_numerator=args.proof_budget_cpu_numerator,
            cpu_denominator=args.proof_budget_cpu_denominator,
            cpu_imag=args.proof_budget_cpu_imag,
            replay_epsilon=args.proof_budget_replay_epsilon,
        )
        candidate = budget["candidate"]
        stationarity = budget["stationarity"]
        print(f"proof_budget_ratio={candidate['cpu_replay_ratio']:.17g}", flush=True)
        print(f"proof_budget_gpu_cpu_difference={candidate['gpu_cpu_difference']:.3e}", flush=True)
        remaining_gain = stationarity["remaining_gain_estimate"]
        if remaining_gain is not None:
            print(f"proof_budget_remaining_gain_estimate={remaining_gain:.17g}", flush=True)
        if args.proof_budget_md is not None:
            print(f"proof_budget_md={args.proof_budget_md}", flush=True)
        if args.proof_budget_output_json is not None:
            print(f"proof_budget_output_json={args.proof_budget_output_json}", flush=True)
        return

    if args.direct_expand_parent_json is not None:
        required = {
            "--direct-expand-parent-label": args.direct_expand_parent_label,
            "--direct-expand-rank-start": args.direct_expand_rank_start,
            "--direct-expand-rank-end": args.direct_expand_rank_end,
            "--direct-expand-split-dir": args.direct_expand_split_dir,
            "--direct-expand-output-json": args.direct_expand_output_json,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError("--direct-expand-parent-json also requires " + ", ".join(missing))
        assert args.direct_expand_parent_label is not None
        assert args.direct_expand_rank_start is not None
        assert args.direct_expand_rank_end is not None
        assert args.direct_expand_split_dir is not None
        assert args.direct_expand_output_json is not None
        expanded = expand_parent_residual_json(
            source_json=args.direct_expand_parent_json,
            parent_label=args.direct_expand_parent_label,
            child_rank_start=args.direct_expand_rank_start,
            child_rank_end=args.direct_expand_rank_end,
            split_dir=args.direct_expand_split_dir,
            residual_cache_path=args.direct_expand_residual_cache,
            output_json=args.direct_expand_output_json,
            csv_path=args.csv_path,
            base_s2=args.base_s2,
        )
        print(f"direct_expand_output_json={args.direct_expand_output_json}", flush=True)
        print(f"direct_expand_group_count={expanded['group_count']}", flush=True)
        print(f"direct_expand_parent_scale={expanded['expanded_parent_scale']:.17g}", flush=True)
        return

    if args.derivative_audit_json is not None:
        replay_summary = json.loads(args.derivative_audit_json.read_text(encoding="utf-8"))
        coeffs = np.asarray(replay_summary["polynomial_coefficients"], dtype=np.float64)
        exponents = [tuple(item) for item in replay_summary["monomial_exponents"]]
        free_scales = np.asarray(replay_summary["free_scales"], dtype=np.float64)
        scalar_scale = float(replay_summary["scalar_best_scale"])
        delta_x = np.asarray([row["delta_x"] for row in replay_summary["block_rows"]], dtype=np.float64)
        delta_d = np.asarray([row["delta_d"] for row in replay_summary["block_rows"]], dtype=np.float64)

        free_value, free_gradient, free_hessian = ratio_value_gradient_hessian(
            free_scales, coeffs, exponents, delta_x, delta_d
        )
        free_eigs = np.linalg.eigvalsh(free_hessian)
        z2 = free_scales * free_scales
        x_factor = 1.0 + float(delta_x @ z2)
        d_factor = 1.0 + float(delta_d @ z2)
        denominator = x_factor * math.sqrt(d_factor)
        inverse_denominator = 1.0 / denominator
        phi_gradient = 2.0 * delta_x * free_scales / x_factor + delta_d * free_scales / d_factor
        phi_hessian = np.diag(2.0 * delta_x / x_factor + delta_d / d_factor)
        phi_hessian -= np.outer(2.0 * delta_x * free_scales, 2.0 * delta_x * free_scales) / (x_factor * x_factor)
        phi_hessian -= 2.0 * np.outer(delta_d * free_scales, delta_d * free_scales) / (d_factor * d_factor)
        hessian_weight_fro_sum = 0.0
        gradient_weight_l2_sum = 0.0
        value_weight_abs_sum = 0.0
        for powers in exponents:
            monomial, monomial_gradient, monomial_hessian = monomial_value_gradient_hessian(free_scales, powers)
            value_weight_abs_sum += abs(inverse_denominator * monomial)
            gradient_contribution = inverse_denominator * (monomial_gradient - monomial * phi_gradient)
            hessian_contribution = inverse_denominator * (
                monomial_hessian
                - np.outer(monomial_gradient, phi_gradient)
                - np.outer(phi_gradient, monomial_gradient)
                + monomial * (np.outer(phi_gradient, phi_gradient) - phi_hessian)
            )
            gradient_weight_l2_sum += float(np.linalg.norm(gradient_contribution))
            hessian_weight_fro_sum += float(np.linalg.norm(hessian_contribution, ord="fro"))
        curvature_gap = max(0.0, -float(free_eigs[-1]))
        allowed_coeff_abs_for_hessian = curvature_gap / hessian_weight_fro_sum if hessian_weight_fro_sum else math.inf
        scalar_point = np.full(len(free_scales), scalar_scale, dtype=np.float64)
        scalar_value, scalar_gradient, _ = ratio_value_gradient_hessian(
            scalar_point, coeffs, exponents, delta_x, delta_d
        )
        scalar_direction = np.ones(len(free_scales), dtype=np.float64)
        scalar_direction /= np.linalg.norm(scalar_direction)
        transverse_gradient = scalar_gradient - scalar_direction * float(scalar_gradient @ scalar_direction)

        print(f"audit_json={args.derivative_audit_json}", flush=True)
        print(f"analytic_free_ratio={free_value:.17g}", flush=True)
        print(f"analytic_free_gradient_norm={np.linalg.norm(free_gradient):.17g}", flush=True)
        print(f"analytic_free_gradient_max_abs={np.max(np.abs(free_gradient)):.17g}", flush=True)
        print(f"analytic_free_hessian_min_eig={free_eigs[0]:.17g}", flush=True)
        print(f"analytic_free_hessian_max_eig={free_eigs[-1]:.17g}", flush=True)
        print(f"analytic_scalar_ratio={scalar_value:.17g}", flush=True)
        print(f"analytic_scalar_transverse_gradient_norm={np.linalg.norm(transverse_gradient):.17g}", flush=True)
        print(f"interior_min_scale={np.min(free_scales):.17g}", flush=True)
        print(f"interior_margin_to_scale_max={args.scale_max - np.max(free_scales):.17g}", flush=True)
        print(f"coefficient_value_abs_weight_sum={value_weight_abs_sum:.17g}", flush=True)
        print(f"coefficient_gradient_l2_weight_sum={gradient_weight_l2_sum:.17g}", flush=True)
        print(f"coefficient_hessian_fro_weight_sum={hessian_weight_fro_sum:.17g}", flush=True)
        print(f"allowed_coeff_abs_for_hessian_sign={allowed_coeff_abs_for_hessian:.17g}", flush=True)
        print(f"analytic_free_hessian_eigs={' '.join(f'{value:.9e}' for value in free_eigs)}", flush=True)

        if args.derivative_audit_md is not None:
            args.derivative_audit_md.parent.mkdir(parents=True, exist_ok=True)
            args.derivative_audit_md.write_text(
                "\n".join(
                    [
                        "# C* W30 Analytic Derivative Audit",
                        "",
                        "This is a tracking artifact, not manuscript text.  It differentiates the saved",
                        "W30 fitted cubic numerator analytically and combines it with the exact",
                        "quadratic denominator formula.",
                        "",
                        "## Inputs",
                        "",
                        f"- source JSON: `{args.derivative_audit_json}`",
                        f"- dimension: `{len(free_scales)}`",
                        f"- monomial coefficients: `{len(coeffs)}`",
                        "",
                        "## Free Branch",
                        "",
                        f"- analytic ratio: `{free_value:.17g}`",
                        f"- gradient norm: `{np.linalg.norm(free_gradient):.17g}`",
                        f"- max absolute gradient component: `{np.max(np.abs(free_gradient)):.17g}`",
                        f"- Hessian minimum eigenvalue: `{free_eigs[0]:.17g}`",
                        f"- Hessian maximum eigenvalue: `{free_eigs[-1]:.17g}`",
                        f"- minimum scale: `{np.min(free_scales):.17g}`",
                        f"- margin to scale cap `{args.scale_max}`: `{args.scale_max - np.max(free_scales):.17g}`",
                        "",
                        "## Coefficient Sensitivity",
                        "",
                        "These are conservative linear weights for a uniform absolute perturbation",
                        "of every fitted numerator coefficient at the audited point.",
                        "",
                        f"- ratio value absolute-weight sum: `{value_weight_abs_sum:.17g}`",
                        f"- gradient l2-weight sum: `{gradient_weight_l2_sum:.17g}`",
                        f"- Hessian Frobenius-weight sum: `{hessian_weight_fro_sum:.17g}`",
                        f"- coefficient error allowed by Hessian sign: `{allowed_coeff_abs_for_hessian:.17g}`",
                        "",
                        "## Scalar Branch Check",
                        "",
                        f"- analytic scalar ratio: `{scalar_value:.17g}`",
                        f"- scalar transverse gradient norm: `{np.linalg.norm(transverse_gradient):.17g}`",
                        "",
                        "## Interpretation",
                        "",
                        "The analytic Hessian is the right local-curvature diagnostic for the fitted",
                        "polynomial model.  It replaces the fragile finite-difference sign check.",
                        "This does not by itself close the paper theorem; it is the active-face",
                        "input for the interval Schur/KKT certificate.",
                        "",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"derivative_audit_md={args.derivative_audit_md}", flush=True)
        return

    layout_id = f"w{args.rank_window}" if args.rank_window is not None else args.layout
    blocks = default_blocks(args.results_dir, args.layout, args.rank_window)
    all_selected_shells: list[int] = []
    block_shells: list[list[int]] = []
    for block in blocks:
        shells, _, _, _, _ = selected_shells_from_optimizer(
            args.csv_path, args.base_s2, block.rank_start, block.rank_count
        )
        block_shells.append(shells)
        all_selected_shells.extend(shells)
        missing = [path for path in block.cache_parts if not path.exists()]
        if missing:
            raise FileNotFoundError(f"missing direction cache parts for {block.label}: {missing[:5]}")

    selected_shell_set = set(all_selected_shells)
    cap_s2 = max(selected_shell_set)
    active_pos = positive_modes(args.base_s2)
    common_selected_pos, common_selected_full = modes_on_shells(selected_shell_set, cap_s2)
    common_index = {mode: index for index, mode in enumerate(common_selected_pos)}
    all_pos = active_pos + common_selected_pos
    e1, e2, _ = basis_arrays(all_pos)
    active_coeffs = load_base_coeffs(args.base_s2, args.coeff_dir, args.base_coeffs)
    active_padded = np.vstack([active_coeffs, np.zeros((len(common_selected_pos), 4), dtype=np.float64)])

    block_directions: list[np.ndarray] = []
    block_linear: list[float] = []
    block_dx: list[float] = []
    block_dd: list[float] = []
    x2 = d2 = None
    block_rows: list[dict[str, object]] = []
    for block, shells in zip(blocks, block_shells):
        block_selected_pos, _ = modes_on_shells(set(shells), max(shells))
        direction, part_x2, part_d2, linear, delta_x, delta_d, _ = load_direction_cache_parts(
            list(block.cache_parts), shells, block_selected_pos
        )
        if x2 is None:
            x2 = part_x2
            d2 = part_d2
        elif abs(x2 - part_x2) > 1e-6 or abs(d2 - part_d2) > 1e-4:
            raise ValueError("block cache invariant mismatch")
        common_direction = np.zeros((len(common_selected_pos), 4), dtype=np.float64)
        for local_index, mode in enumerate(block_selected_pos):
            common_direction[common_index[mode]] = direction[local_index]
        block_directions.append(common_direction)
        block_linear.append(linear)
        block_dx.append(delta_x)
        block_dd.append(delta_d)
        block_rows.append(
            {
                "label": block.label,
                "rank_start": block.rank_start,
                "rank_end": block.rank_start + block.rank_count - 1,
                "shell_count": len(shells),
                "linear": linear,
                "delta_x": delta_x,
                "delta_d": delta_d,
                "cache_count": len(block.cache_parts),
            }
        )

    assert x2 is not None and d2 is not None
    max_coord = max(max(abs(component) for component in mode) for mode in (active_pos + common_selected_full))
    alias_safe = args.fft_grid > 3 * max_coord
    if not alias_safe:
        raise ValueError(f"fft grid {args.fft_grid} is not alias-safe for max coordinate {max_coord}")

    print(f"blocks={len(blocks)} selected_shells={len(selected_shell_set)} positive_modes={len(all_pos)}", flush=True)
    print(
        f"fft_grid={args.fft_grid} max_coord={max_coord} alias_safe={alias_safe} "
        f"backend={args.backend} workers={args.fft_workers} torch_precision={args.torch_precision}",
        flush=True,
    )
    print("building coefficient grids", flush=True)
    active_grids = coefficient_grids(all_pos, active_padded, e1, e2, args.fft_grid)
    block_grids: list[list[np.ndarray]] = []
    for direction in block_directions:
        padded = np.vstack([np.zeros_like(active_coeffs), direction])
        block_grids.append(coefficient_grids(all_pos, padded, e1, e2, args.fft_grid))
    if args.backend == "torch-cuda":
        evaluator = TorchCudaFftNumerator(active_grids, block_grids, x2, d2, args.fft_grid, args.torch_precision)
    else:
        evaluator = FftNumerator(active_grids, block_grids, x2, d2, args.fft_grid, args.backend, args.fft_workers)

    if args.replay_json is not None:
        replay_summary = json.loads(args.replay_json.read_text(encoding="utf-8"))
        free_scales = np.asarray(replay_summary["free_scales"], dtype=np.float64)
        scalar_scale = float(replay_summary["scalar_best_scale"])
        delta_x = np.asarray(block_dx, dtype=np.float64)
        delta_d = np.asarray(block_dd, dtype=np.float64)
        replay_points = [
            ("scalar", np.full(len(blocks), scalar_scale, dtype=np.float64)),
            ("free", free_scales),
        ]
        if hasattr(evaluator, "torch"):
            evaluator.torch.cuda.synchronize()
        for name, point in replay_points:
            numerator, imag = evaluator(point)
            z2 = point * point
            denominator = (1.0 + float(delta_x @ z2)) * math.sqrt(1.0 + float(delta_d @ z2))
            ratio = numerator / denominator
            print(
                f"replay_point={name} numerator={numerator:.17g} denominator={denominator:.17g} "
                f"ratio={ratio:.17g} imag={imag:.3e}",
                flush=True,
            )
        if hasattr(evaluator, "torch"):
            evaluator.torch.cuda.synchronize()
        return

    if args.line_audit_json is not None:
        replay_summary = json.loads(args.line_audit_json.read_text(encoding="utf-8"))
        coeffs = np.asarray(replay_summary["polynomial_coefficients"], dtype=np.float64)
        exponents = [tuple(item) for item in replay_summary["monomial_exponents"]]
        free_scales = np.asarray(replay_summary["free_scales"], dtype=np.float64)
        delta_x = np.asarray(block_dx, dtype=np.float64)
        delta_d = np.asarray(block_dd, dtype=np.float64)
        steps = [float(item.strip()) for item in args.line_audit_steps.split(",") if item.strip()]
        if not steps:
            raise ValueError("--line-audit-steps must contain at least one positive step")
        if any(step <= 0.0 for step in steps):
            raise ValueError("--line-audit-steps must be positive")

        analytic_value, _, analytic_hessian = ratio_value_gradient_hessian(
            free_scales, coeffs, exponents, delta_x, delta_d
        )
        analytic_eigs, analytic_vectors = np.linalg.eigh(analytic_hessian)

        def replay_ratio(point: np.ndarray) -> tuple[float, float]:
            numerator, imag = evaluator(point)
            z2 = point * point
            denominator = (1.0 + float(delta_x @ z2)) * math.sqrt(1.0 + float(delta_d @ z2))
            return numerator / denominator, imag

        if hasattr(evaluator, "torch"):
            evaluator.torch.cuda.synchronize()
        direct_base, direct_base_imag = replay_ratio(free_scales)
        rows: list[dict[str, object]] = []
        top_count = min(max(1, args.line_audit_top_eigs), len(free_scales))
        for rank in range(1, top_count + 1):
            eig_index = len(free_scales) - rank
            direction = analytic_vectors[:, eig_index]
            eig_value = float(analytic_eigs[eig_index])
            for step in steps:
                plus = free_scales + step * direction
                minus = free_scales - step * direction
                if np.min(plus) < 0.0 or np.min(minus) < 0.0:
                    raise ValueError(f"line audit step {step} leaves nonnegative scale domain")
                if np.max(plus) > args.scale_max or np.max(minus) > args.scale_max:
                    raise ValueError(f"line audit step {step} exceeds --scale-max")
                direct_plus, imag_plus = replay_ratio(plus)
                direct_minus, imag_minus = replay_ratio(minus)
                direct_curvature = (direct_plus - 2.0 * direct_base + direct_minus) / (step * step)
                direct_slope = (direct_plus - direct_minus) / (2.0 * step)
                poly_plus = ratio_from_poly(plus, coeffs, exponents, delta_x, delta_d)
                poly_minus = ratio_from_poly(minus, coeffs, exponents, delta_x, delta_d)
                poly_curvature = (poly_plus - 2.0 * analytic_value + poly_minus) / (step * step)
                rows.append(
                    {
                        "direction_rank": rank,
                        "eigenvalue": eig_value,
                        "step": step,
                        "direct_curvature": float(direct_curvature),
                        "poly_curvature": float(poly_curvature),
                        "curvature_error": float(direct_curvature - poly_curvature),
                        "direct_slope": float(direct_slope),
                        "max_imag": float(max(abs(direct_base_imag), abs(imag_plus), abs(imag_minus))),
                    }
                )
                print(
                    f"line_rank={rank} step={step:.6g} eig={eig_value:.17g} "
                    f"direct_curvature={direct_curvature:.17g} poly_curvature={poly_curvature:.17g} "
                    f"curvature_error={direct_curvature - poly_curvature:.3e} direct_slope={direct_slope:.3e} "
                    f"max_imag={max(abs(direct_base_imag), abs(imag_plus), abs(imag_minus)):.3e}",
                    flush=True,
                )
        if hasattr(evaluator, "torch"):
            evaluator.torch.cuda.synchronize()

        print(f"line_audit_base_direct_ratio={direct_base:.17g}", flush=True)
        print(f"line_audit_base_analytic_ratio={analytic_value:.17g}", flush=True)
        print(f"line_audit_base_error={direct_base - analytic_value:.3e}", flush=True)

        if args.line_audit_md is not None:
            args.line_audit_md.parent.mkdir(parents=True, exist_ok=True)
            lines = [
                "# C* W30 Direct FFT Line-Curvature Audit",
                "",
                "This is a tracking artifact, not manuscript text.  It replays the actual",
                "alias-safe FFT trilinear ratio along the weakest analytic Hessian directions",
                "of the W30 fitted-ratio candidate.",
                "",
                "## Inputs",
                "",
                f"- source JSON: `{args.line_audit_json}`",
                f"- backend: `{args.backend}`",
                f"- fft grid: `{args.fft_grid}`",
                f"- base direct ratio: `{direct_base:.17g}`",
                f"- base analytic ratio: `{analytic_value:.17g}`",
                f"- base replay minus analytic: `{direct_base - analytic_value:.3e}`",
                "",
                "## Line Curvatures",
                "",
                "| rank from top | analytic eig | step | direct curvature | fitted curvature | error | direct slope | max imag |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for row in rows:
                lines.append(
                    f"| `{row['direction_rank']}` "
                    f"| `{row['eigenvalue']:.17g}` "
                    f"| `{row['step']:.6g}` "
                    f"| `{row['direct_curvature']:.17g}` "
                    f"| `{row['poly_curvature']:.17g}` "
                    f"| `{row['curvature_error']:.3e}` "
                    f"| `{row['direct_slope']:.3e}` "
                    f"| `{row['max_imag']:.3e}` |"
                )
            lines.extend(
                [
                    "",
                    "## Interpretation",
                    "",
                    "The direct FFT line audit is independent of the polynomial derivative",
                    "calculation except for the choice of directions.  Agreement here supports",
                    "using the analytic Hessian as the active-face object for the interval",
                    "Schur/KKT certificate.",
                    "",
                ]
            )
            args.line_audit_md.write_text("\n".join(lines), encoding="utf-8")
            print(f"line_audit_md={args.line_audit_md}", flush=True)
        return

    if args.split_audit_json is not None:
        replay_summary = json.loads(args.split_audit_json.read_text(encoding="utf-8"))
        free_scales = np.asarray(replay_summary["free_scales"], dtype=np.float64)
        if len(free_scales) != len(blocks):
            raise ValueError(
                f"split audit JSON has {len(free_scales)} scales, but current layout has {len(blocks)} blocks"
            )
        split_steps = [float(item.strip()) for item in args.split_audit_steps.split(",") if item.strip()]
        if not split_steps:
            raise ValueError("--split-audit-steps must contain at least one positive step")
        if any(step <= 0.0 for step in split_steps):
            raise ValueError("--split-audit-steps must be positive")
        if args.split_audit_window <= 0 or args.split_audit_window % 10 != 0:
            raise ValueError("--split-audit-window must be a positive multiple of 10")
        if args.split_audit_max_rank > 700:
            raise ValueError("--split-audit-max-rank cannot exceed 700 with the current cached rank windows")

        candidate_grids = [component.copy() for component in active_grids]
        for scale, block_grid in zip(free_scales, block_grids):
            for component in range(3):
                candidate_grids[component] += float(scale) * block_grid[component]

        delta_x = np.asarray(block_dx, dtype=np.float64)
        delta_d = np.asarray(block_dd, dtype=np.float64)
        x_factor0 = 1.0 + float(delta_x @ (free_scales * free_scales))
        d_factor0 = 1.0 + float(delta_d @ (free_scales * free_scales))

        if args.backend == "torch-cuda":
            split_evaluator = TorchCudaSingleDirectionFftNumerator(
                candidate_grids, x2, d2, args.fft_grid, args.torch_precision
            )
            base_numerator, base_imag = split_evaluator.evaluate_base()
            if hasattr(split_evaluator, "torch"):
                split_evaluator.torch.cuda.synchronize()
        else:
            split_evaluator = SingleDirectionFftNumerator(
                candidate_grids, x2, d2, args.fft_grid, args.backend, args.fft_workers
            )
            base_numerator, base_imag = split_evaluator.evaluate()
        base_ratio = base_numerator / (x_factor0 * math.sqrt(d_factor0))

        rows: list[dict[str, object]] = []
        summary_by_split: list[dict[str, object]] = []
        for lo in range(1, args.split_audit_max_rank + 1, args.split_audit_window):
            hi = min(lo + args.split_audit_window - 1, args.split_audit_max_rank)
            if hi % 10 != 0:
                hi -= hi % 10
            if hi < lo:
                continue
            parent_index = None
            for index, block in enumerate(blocks):
                parent_hi = block.rank_start + block.rank_count - 1
                if block.rank_start <= lo and hi <= parent_hi:
                    parent_index = index
                    break
            if parent_index is None:
                raise ValueError(f"no parent block contains split rank window {lo}..{hi}")

            split_shells, _, _, _, _ = selected_shells_from_optimizer(
                args.csv_path, args.base_s2, lo, hi - lo + 1
            )
            split_pos, _ = modes_on_shells(set(split_shells), max(split_shells))
            split_direction, part_x2, part_d2, linear, split_dx, split_dd, _ = load_direction_cache_parts(
                cache_paths_for_range(args.results_dir, lo, hi), split_shells, split_pos
            )
            if abs(x2 - part_x2) > 1e-6 or abs(d2 - part_d2) > 1e-4:
                raise ValueError(f"split cache invariant mismatch for {lo}..{hi}")

            common_direction = np.zeros((len(common_selected_pos), 4), dtype=np.float64)
            for local_index, mode in enumerate(split_pos):
                common_direction[common_index[mode]] = split_direction[local_index]
            padded = np.vstack([np.zeros_like(active_coeffs), common_direction])
            direction_grids = coefficient_grids(all_pos, padded, e1, e2, args.fft_grid)
            direction_tensors = None
            if args.backend == "torch-cuda":
                direction_tensors = split_evaluator.direction_tensors(direction_grids)

            parent_scale = float(free_scales[parent_index])
            parent_label = blocks[parent_index].label
            step_rows: list[dict[str, object]] = []
            for step in split_steps:
                if parent_scale - step < 0.0 or parent_scale + step > args.scale_max:
                    raise ValueError(f"split step {step} leaves scale bounds for parent {parent_label}")

                def ratio_at(epsilon: float) -> tuple[float, float]:
                    if args.backend == "torch-cuda":
                        numerator, imag = split_evaluator.evaluate_direction(direction_tensors, epsilon)
                    else:
                        numerator, imag = split_evaluator.evaluate(direction_grids, epsilon)
                    x_factor = x_factor0 + 2.0 * parent_scale * epsilon * split_dx + epsilon * epsilon * split_dx
                    d_factor = d_factor0 + 2.0 * parent_scale * epsilon * split_dd + epsilon * epsilon * split_dd
                    return numerator / (x_factor * math.sqrt(d_factor)), imag

                plus_ratio, plus_imag = ratio_at(step)
                minus_ratio, minus_imag = ratio_at(-step)
                slope = (plus_ratio - minus_ratio) / (2.0 * step)
                curvature = (plus_ratio - 2.0 * base_ratio + minus_ratio) / (step * step)
                row = {
                    "split": f"r{lo:03d}_{hi:03d}",
                    "parent": parent_label,
                    "step": step,
                    "slope": float(slope),
                    "curvature": float(curvature),
                    "plus_ratio": float(plus_ratio),
                    "minus_ratio": float(minus_ratio),
                    "max_imag": float(max(abs(base_imag), abs(plus_imag), abs(minus_imag))),
                    "linear": float(linear),
                    "delta_x": float(split_dx),
                    "delta_d": float(split_dd),
                }
                rows.append(row)
                step_rows.append(row)
                print(
                    f"split={row['split']} parent={parent_label} step={step:.6g} "
                    f"slope={slope:.17g} curvature={curvature:.17g} "
                    f"plus={plus_ratio:.17g} minus={minus_ratio:.17g} "
                    f"max_imag={row['max_imag']:.3e}",
                    flush=True,
                )
            representative = max(step_rows, key=lambda item: abs(float(item["slope"])))
            summary_by_split.append(representative)
            if args.backend == "torch-cuda" and hasattr(split_evaluator, "torch"):
                split_evaluator.torch.cuda.empty_cache()

        summary_sorted = sorted(summary_by_split, key=lambda item: abs(float(item["slope"])), reverse=True)
        print(f"split_audit_base_ratio={base_ratio:.17g}", flush=True)
        print(f"split_audit_rows={len(rows)}", flush=True)
        print(
            "split_audit_top_abs_slopes="
            + " ".join(
                f"{row['split']}:{float(row['slope']):.6e}" for row in summary_sorted[: min(12, len(summary_sorted))]
            ),
            flush=True,
        )

        if args.split_audit_md is not None:
            args.split_audit_md.parent.mkdir(parents=True, exist_ok=True)
            lines = [
                "# C* W30 Rank-Split Inactive Audit",
                "",
                "This is a tracking artifact, not manuscript text.  It replays direct",
                "alias-safe FFT line slopes for cached rank-window split directions inside",
                "the W30 block-scaled branch.",
                "",
                "## Inputs",
                "",
                f"- source JSON: `{args.split_audit_json}`",
                f"- backend: `{args.backend}`",
                f"- fft grid: `{args.fft_grid}`",
                f"- split window: `{args.split_audit_window}`",
                f"- max audited rank: `{args.split_audit_max_rank}`",
                f"- base direct ratio: `{base_ratio:.17g}`",
                "",
                "## Largest Split Slopes",
                "",
                "| split | parent | step | slope | curvature | plus ratio | minus ratio | max imag |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
            for row in summary_sorted[: min(30, len(summary_sorted))]:
                lines.append(
                    f"| `{row['split']}` "
                    f"| `{row['parent']}` "
                    f"| `{float(row['step']):.6g}` "
                    f"| `{float(row['slope']):.17g}` "
                    f"| `{float(row['curvature']):.17g}` "
                    f"| `{float(row['plus_ratio']):.17g}` "
                    f"| `{float(row['minus_ratio']):.17g}` "
                    f"| `{float(row['max_imag']):.3e}` |"
                )
            lines.extend(
                [
                    "",
                    "## All Rows",
                    "",
                    "| split | parent | step | slope | curvature | plus ratio | minus ratio | max imag |",
                    "|---|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in rows:
                lines.append(
                    f"| `{row['split']}` "
                    f"| `{row['parent']}` "
                    f"| `{float(row['step']):.6g}` "
                    f"| `{float(row['slope']):.17g}` "
                    f"| `{float(row['curvature']):.17g}` "
                    f"| `{float(row['plus_ratio']):.17g}` "
                    f"| `{float(row['minus_ratio']):.17g}` "
                    f"| `{float(row['max_imag']):.3e}` |"
                )
            lines.extend(
                [
                    "",
                    "## Interpretation",
                    "",
                    "This audit tests whether the W30 branch is stationary against cached",
                    "rank-window split directions.  Nonzero split slopes identify the next",
                    "finite-dimensional complementarity target; they are not a full inactive",
                    "proof by themselves.",
                    "",
                ]
            )
            args.split_audit_md.write_text("\n".join(lines), encoding="utf-8")
            print(f"split_audit_md={args.split_audit_md}", flush=True)
        return

    if args.direct_optimize_json is not None:
        if args.backend != "torch-cuda":
            raise ValueError("--direct-optimize-json currently requires --backend torch-cuda")
        source_summary = json.loads(args.direct_optimize_json.read_text(encoding="utf-8"))
        source_scales = None
        source_scale_by_label = None
        if "free_scales" in source_summary:
            source_scales = np.asarray(source_summary["free_scales"], dtype=np.float64)
            if len(source_scales) != len(blocks):
                raise ValueError(
                    f"direct optimize JSON has {len(source_scales)} scales, but current layout has {len(blocks)} blocks"
                )
        elif "scales" in source_summary and "group_rows" in source_summary:
            source_scale_by_label = {
                str(row["label"]): float(scale)
                for row, scale in zip(source_summary["group_rows"], source_summary["scales"])
            }
        else:
            raise ValueError("direct optimize source JSON must contain either free_scales or scales/group_rows")
        group_specs: list[tuple[str, int, int, tuple[Path, ...], float]] = []
        if args.direct_optimize_use_source_groups:
            if "scales" not in source_summary or "group_rows" not in source_summary:
                raise ValueError("--direct-optimize-use-source-groups requires scales/group_rows in source JSON")
            for row, scale in zip(source_summary["group_rows"], source_summary["scales"]):
                rank_start = int(row["rank_start"])
                rank_end = int(row["rank_end"])
                group_specs.append(
                    (
                        str(row["label"]),
                        rank_start,
                        rank_end - rank_start + 1,
                        cache_parts_from_group_row(row, args.results_dir),
                        float(scale),
                    )
                )
            removed_bound_coordinates = list(source_summary.get("removed_bound_coordinates", []))
        else:
            removed_bound_coordinates = []
            if args.direct_optimize_window <= 0 or args.direct_optimize_window % 10 != 0:
                raise ValueError("--direct-optimize-window must be a positive multiple of 10")
            if args.direct_optimize_max_rank > 700:
                raise ValueError("--direct-optimize-max-rank cannot exceed 700 with current cached rank windows")

            for parent_index, block in enumerate(blocks):
                parent_hi = block.rank_start + block.rank_count - 1

                def initial_scale_for(label: str) -> float:
                    if source_scale_by_label is not None:
                        if label not in source_scale_by_label:
                            raise ValueError(f"source direct scales do not contain group {label}")
                        return source_scale_by_label[label]
                    assert source_scales is not None
                    return float(source_scales[parent_index])

                if block.rank_start <= args.direct_optimize_max_rank and parent_hi <= 700:
                    split_hi = min(parent_hi, args.direct_optimize_max_rank)
                    lo = block.rank_start
                    while lo <= split_hi:
                        hi = min(lo + args.direct_optimize_window - 1, split_hi)
                        if hi % 10 != 0:
                            hi -= hi % 10
                        if hi < lo:
                            raise ValueError(f"invalid direct optimize split at rank {lo}")
                        group_specs.append(
                            (
                                f"r{lo:03d}_{hi:03d}",
                                lo,
                                hi - lo + 1,
                                cache_paths_for_range(args.results_dir, lo, hi),
                                initial_scale_for(f"r{lo:03d}_{hi:03d}"),
                            )
                        )
                        lo = hi + 1
                    if split_hi < parent_hi:
                        lo = split_hi + 1
                        label = f"r{lo:03d}_{parent_hi:03d}"
                        group_specs.append((label, lo, parent_hi - lo + 1, block.cache_parts, initial_scale_for(label)))
                else:
                    group_specs.append(
                        (
                            block.label,
                            block.rank_start,
                            block.rank_count,
                            block.cache_parts,
                            initial_scale_for(block.label),
                        )
                    )

        selected_indices: list[int] = []
        group_ids: list[int] = []
        direction_values: list[np.ndarray] = []
        group_delta_x: list[float] = []
        group_delta_d: list[float] = []
        group_rows: list[dict[str, object]] = []
        covered_common: set[int] = set()
        initial_scales: list[float] = []

        for group_index, (label, rank_start, rank_count, cache_parts, initial_scale) in enumerate(group_specs):
            group_shells, _, _, _, _ = selected_shells_from_optimizer(
                args.csv_path, args.base_s2, rank_start, rank_count
            )
            group_pos, _ = modes_on_shells(set(group_shells), max(group_shells))
            direction, part_x2, part_d2, linear, group_dx, group_dd, _ = load_direction_cache_parts(
                cache_parts, group_shells, group_pos
            )
            if abs(x2 - part_x2) > 1e-6 or abs(d2 - part_d2) > 1e-4:
                raise ValueError(f"direct optimize cache invariant mismatch for {label}")
            for local_index, mode in enumerate(group_pos):
                common_i = common_index[mode]
                if common_i in covered_common:
                    raise ValueError(f"mode {mode} appears in multiple direct optimize groups")
                covered_common.add(common_i)
                selected_indices.append(len(active_pos) + common_i)
                group_ids.append(group_index)
                direction_values.append(direction[local_index].copy())
            group_delta_x.append(float(group_dx))
            group_delta_d.append(float(group_dd))
            initial_scales.append(initial_scale)
            group_rows.append(
                {
                    "label": label,
                    "rank_start": rank_start,
                    "rank_end": rank_start + rank_count - 1,
                    "rank_count": rank_count,
                    "shell_count": len(group_shells),
                    "linear": float(linear),
                    "delta_x": float(group_dx),
                    "delta_d": float(group_dd),
                    "initial_scale": initial_scale,
                    "cache_count": len(cache_parts),
                    "cache_parts": [str(path) for path in cache_parts],
                }
            )

        expected_common = {common_index[mode] for mode in common_selected_pos}
        missing_common = expected_common - covered_common
        if missing_common:
            allowed_missing_common: set[int] = set()
            for removed in removed_bound_coordinates:
                rank_start = int(removed["rank_start"])
                rank_end = int(removed.get("rank_end", rank_start))
                group_shells, _, _, _, _ = selected_shells_from_optimizer(
                    args.csv_path, args.base_s2, rank_start, rank_end - rank_start + 1
                )
                group_pos, _ = modes_on_shells(set(group_shells), max(group_shells))
                for mode in group_pos:
                    if mode in common_index:
                        allowed_missing_common.add(common_index[mode])
            unexpected_missing = missing_common - allowed_missing_common
            if unexpected_missing:
                raise ValueError(
                    "direct optimize groups do not cover all selected modes; "
                    f"missing {len(missing_common)}, unexpected {len(unexpected_missing)}"
                )

        direct_problem = TorchCudaDirectScaleProblem(
            all_pos=all_pos,
            base_coeffs_real=active_padded,
            selected_indices=np.asarray(selected_indices, dtype=np.int64),
            group_ids=np.asarray(group_ids, dtype=np.int64),
            direction_values=np.asarray(direction_values, dtype=np.float64),
            e1=e1,
            e2=e2,
            delta_x=np.asarray(group_delta_x, dtype=np.float64),
            delta_d=np.asarray(group_delta_d, dtype=np.float64),
            x2=x2,
            d2=d2,
            grid_size=args.fft_grid,
            precision=args.torch_precision,
        )

        initial = np.asarray(initial_scales, dtype=np.float64)
        call_counter = {"count": 0}
        initial_value, initial_gradient = direct_problem.value_and_grad(initial)
        print(f"direct_initial_ratio={initial_value:.17g}", flush=True)
        print(f"direct_initial_grad_norm={np.linalg.norm(initial_gradient):.17g}", flush=True)
        print(f"direct_initial_grad_max_abs={np.max(np.abs(initial_gradient)):.17g}", flush=True)

        if args.direct_optimize_method == "scipy":
            def objective(scales_np: np.ndarray) -> tuple[float, np.ndarray]:
                value, gradient = direct_problem.value_and_grad(np.asarray(scales_np, dtype=np.float64))
                call_counter["count"] += 1
                if call_counter["count"] == 1 or call_counter["count"] % 5 == 0:
                    print(
                        f"direct_opt_eval={call_counter['count']} ratio={value:.17g} "
                        f"grad_norm={np.linalg.norm(gradient):.6e}",
                        flush=True,
                    )
                return -args.direct_optimize_objective_scale * value, -args.direct_optimize_objective_scale * gradient

            result = minimize(
                objective,
                initial,
                jac=True,
                method="L-BFGS-B",
                bounds=[(0.0, args.scale_max)] * len(initial),
                options={
                    "maxiter": args.direct_optimize_maxiter,
                    "gtol": args.direct_optimize_gtol * args.direct_optimize_objective_scale,
                    "ftol": 1e-15,
                    "maxls": 50,
                    "maxcor": 20,
                },
            )
            final_scales = np.asarray(result.x, dtype=np.float64)
            success = bool(result.success)
            status = int(result.status)
            iterations = int(result.nit)
            message = str(result.message)
        else:
            scales = initial.copy()
            value = initial_value
            gradient = initial_gradient.copy()
            success = False
            status = 1
            message = "maximum iterations reached"
            iterations = 0
            for iteration in range(1, args.direct_optimize_maxiter + 1):
                grad_max = float(np.max(np.abs(gradient)))
                grad_norm = float(np.linalg.norm(gradient))
                if grad_max <= args.direct_optimize_gtol:
                    success = True
                    status = 0
                    message = "gradient tolerance reached"
                    break
                direction = gradient / grad_max
                trial_step = args.direct_optimize_step
                accepted = False
                best_trial_value = -math.inf
                best_trial_scales = scales
                for _ in range(args.direct_optimize_backtracks):
                    trial = np.clip(scales + trial_step * direction, 0.0, args.scale_max)
                    trial_value = direct_problem.value(trial)
                    if trial_value > best_trial_value:
                        best_trial_value = trial_value
                        best_trial_scales = trial
                    if trial_value > value + 1e-16:
                        accepted = True
                        scales = trial
                        value = trial_value
                        break
                    trial_step *= 0.5
                if not accepted:
                    message = "line search stalled"
                    status = 2
                    print(
                        f"direct_opt_iter={iteration} stalled best_trial_ratio={best_trial_value:.17g} "
                        f"current_ratio={value:.17g}",
                        flush=True,
                    )
                    break
                value, gradient = direct_problem.value_and_grad(scales)
                iterations = iteration
                print(
                    f"direct_opt_iter={iteration} ratio={value:.17g} gain={value - initial_value:.6e} "
                    f"grad_norm={grad_norm:.6e} grad_max={grad_max:.6e} step={trial_step:.6g}",
                    flush=True,
                )
            else:
                iterations = args.direct_optimize_maxiter
            final_scales = scales
        final_value, final_gradient = direct_problem.value_and_grad(final_scales)
        print(f"direct_final_ratio={final_value:.17g}", flush=True)
        print(f"direct_gain={final_value - initial_value:.17g}", flush=True)
        print(f"direct_final_grad_norm={np.linalg.norm(final_gradient):.17g}", flush=True)
        print(f"direct_final_grad_max_abs={np.max(np.abs(final_gradient)):.17g}", flush=True)
        print(f"direct_success={success} status={status} nit={iterations} message={message}", flush=True)
        print("direct_final_scales=" + " ".join(f"{value:.12g}" for value in final_scales), flush=True)

        output_summary = {
            "source_json": str(args.direct_optimize_json),
            "backend": args.backend,
            "torch_precision": args.torch_precision,
            "fft_grid": args.fft_grid,
            "layout": f"direct_w{args.direct_optimize_window}_through_{args.direct_optimize_max_rank}",
            "group_count": len(group_rows),
            "initial_ratio": float(initial_value),
            "final_ratio": float(final_value),
            "gain": float(final_value - initial_value),
            "initial_gradient_norm": float(np.linalg.norm(initial_gradient)),
            "initial_gradient_max_abs": float(np.max(np.abs(initial_gradient))),
            "final_gradient_norm": float(np.linalg.norm(final_gradient)),
            "final_gradient_max_abs": float(np.max(np.abs(final_gradient))),
            "initial_gradient": [float(value) for value in initial_gradient],
            "final_gradient": [float(value) for value in final_gradient],
            "success": success,
            "status": status,
            "iterations": iterations,
            "function_evaluations": int(call_counter["count"]),
            "message": message,
            "objective_scale": float(args.direct_optimize_objective_scale),
            "method": args.direct_optimize_method,
            "scales": [float(value) for value in final_scales],
            "initial_scales": [float(value) for value in initial],
            "group_rows": group_rows,
        }
        if removed_bound_coordinates:
            output_summary["removed_bound_coordinates"] = removed_bound_coordinates
        if args.direct_optimize_output_json is not None:
            args.direct_optimize_output_json.parent.mkdir(parents=True, exist_ok=True)
            args.direct_optimize_output_json.write_text(json.dumps(output_summary, indent=2) + "\n", encoding="utf-8")
            print(f"direct_output_json={args.direct_optimize_output_json}", flush=True)
        if args.direct_optimize_md is not None:
            args.direct_optimize_md.parent.mkdir(parents=True, exist_ok=True)
            lines = [
                "# C* Direct FFT Split-Scale Optimization",
                "",
                "This is a tracking artifact, not manuscript text.  It optimizes cached",
                "rank-window split scales by differentiating the alias-safe FFT replay on",
                "CUDA, avoiding a high-dimensional cubic polynomial fit.",
                "",
                "## Summary",
                "",
                f"- source JSON: `{args.direct_optimize_json}`",
                f"- backend: `{args.backend}`",
                f"- precision: `{args.torch_precision}`",
                f"- fft grid: `{args.fft_grid}`",
                f"- group count: `{len(group_rows)}`",
                f"- initial ratio: `{initial_value:.17g}`",
                f"- final ratio: `{final_value:.17g}`",
                f"- gain: `{final_value - initial_value:.17g}`",
                f"- initial gradient norm: `{np.linalg.norm(initial_gradient):.17g}`",
                f"- final gradient norm: `{np.linalg.norm(final_gradient):.17g}`",
                f"- method: `{args.direct_optimize_method}`",
                f"- success: `{success}`",
                f"- iterations: `{iterations}`",
                f"- objective scale: `{args.direct_optimize_objective_scale:.17g}`",
                f"- message: `{message}`",
                f"- removed lower-bound coordinates: `{len(removed_bound_coordinates)}`",
                "",
                "## Groups",
                "",
                "| group | ranks | initial scale | final scale | delta X | delta D |",
                "|---|---:|---:|---:|---:|---:|",
            ]
            for row, scale in zip(group_rows, final_scales):
                lines.append(
                    f"| `{row['label']}` "
                    f"| `{row['rank_start']}..{row['rank_end']}` "
                    f"| `{float(row['initial_scale']):.12g}` "
                    f"| `{float(scale):.12g}` "
                    f"| `{float(row['delta_x']):.17g}` "
                    f"| `{float(row['delta_d']):.17g}` |"
                )
            lines.extend(
                [
                    "",
                    "## Interpretation",
                    "",
                    "This gives a direct FFT lower-bound refinement for the cached split",
                    "branch.  It is not an interval proof; CPU replay and interval Schur/KKT",
                    "packaging are still required before manuscript use.",
                    "",
                ]
            )
            args.direct_optimize_md.write_text("\n".join(lines), encoding="utf-8")
            print(f"direct_optimize_md={args.direct_optimize_md}", flush=True)
        return

    if args.direct_hessian_json is not None:
        if args.backend != "torch-cuda":
            raise ValueError("--direct-hessian-json currently requires --backend torch-cuda")
        if args.direct_hessian_step <= 0.0:
            raise ValueError("--direct-hessian-step must be positive")
        if args.direct_hessian_max_coordinates < 0:
            raise ValueError("--direct-hessian-max-coordinates must be nonnegative")
        hessian_summary = json.loads(args.direct_hessian_json.read_text(encoding="utf-8"))
        if "scales" not in hessian_summary or "group_rows" not in hessian_summary:
            raise ValueError("--direct-hessian-json requires a direct optimize JSON with scales/group_rows")
        scales = np.asarray(hessian_summary["scales"], dtype=np.float64)
        group_rows = list(hessian_summary["group_rows"])
        if len(scales) != len(group_rows):
            raise ValueError("direct hessian scales/group_rows length mismatch")

        selected_indices: list[int] = []
        group_ids: list[int] = []
        direction_values: list[np.ndarray] = []
        group_delta_x: list[float] = []
        group_delta_d: list[float] = []
        covered_common: set[int] = set()

        for group_index, row in enumerate(group_rows):
            rank_start = int(row["rank_start"])
            rank_count = int(row["rank_count"])
            rank_end = int(row["rank_end"])
            group_shells, _, _, _, _ = selected_shells_from_optimizer(
                args.csv_path, args.base_s2, rank_start, rank_count
            )
            group_pos, _ = modes_on_shells(set(group_shells), max(group_shells))
            cache_parts = cache_parts_from_group_row(row, args.results_dir)
            direction, part_x2, part_d2, _, group_dx, group_dd, _ = load_direction_cache_parts(
                cache_parts, group_shells, group_pos
            )
            if abs(x2 - part_x2) > 1e-6 or abs(d2 - part_d2) > 1e-4:
                raise ValueError(f"direct hessian cache invariant mismatch for {row['label']}")
            for local_index, mode in enumerate(group_pos):
                common_i = common_index[mode]
                if common_i in covered_common:
                    raise ValueError(f"mode {mode} appears in multiple direct hessian groups")
                covered_common.add(common_i)
                selected_indices.append(len(active_pos) + common_i)
                group_ids.append(group_index)
                direction_values.append(direction[local_index].copy())
            group_delta_x.append(float(group_dx))
            group_delta_d.append(float(group_dd))

        allow_missing_bound_modes = bool(hessian_summary.get("removed_bound_coordinates"))
        if len(covered_common) != len(common_selected_pos) and not allow_missing_bound_modes:
            missing_count = len(common_selected_pos) - len(covered_common)
            raise ValueError(f"direct hessian groups do not cover all selected modes; missing {missing_count}")

        direct_problem = TorchCudaDirectScaleProblem(
            all_pos=all_pos,
            base_coeffs_real=active_padded,
            selected_indices=np.asarray(selected_indices, dtype=np.int64),
            group_ids=np.asarray(group_ids, dtype=np.int64),
            direction_values=np.asarray(direction_values, dtype=np.float64),
            e1=e1,
            e2=e2,
            delta_x=np.asarray(group_delta_x, dtype=np.float64),
            delta_d=np.asarray(group_delta_d, dtype=np.float64),
            x2=x2,
            d2=d2,
            grid_size=args.fft_grid,
            precision=args.torch_precision,
        )

        base_value, base_gradient = direct_problem.value_and_grad(scales)
        ranked_coordinates = np.argsort(np.abs(base_gradient))[::-1]
        step = float(args.direct_hessian_step)
        if args.direct_hessian_free_only:
            candidate_coordinates = [
                index
                for index, scale in enumerate(scales)
                if scale > step and scale < args.scale_max - step
            ]
        elif args.direct_hessian_active_only:
            candidate_coordinates = [index for index, scale in enumerate(scales) if scale > 0.0]
        else:
            candidate_coordinates = list(range(len(scales)))
        if args.direct_hessian_max_coordinates == 0:
            selected_coordinates = candidate_coordinates
        else:
            candidate_set = set(candidate_coordinates)
            selected_coordinates = [
                int(index)
                for index in ranked_coordinates
                if int(index) in candidate_set
            ][: min(args.direct_hessian_max_coordinates, len(candidate_coordinates))]

        hessian_columns = np.empty((len(scales), len(selected_coordinates)), dtype=np.float64)
        column_rows: list[dict[str, object]] = []
        start_time = time.perf_counter()
        print(f"direct_hessian_base_ratio={base_value:.17g}", flush=True)
        print(f"direct_hessian_gradient_norm={np.linalg.norm(base_gradient):.17g}", flush=True)
        print(f"direct_hessian_columns={len(selected_coordinates)}/{len(scales)} step={step:.6g}", flush=True)
        for column_number, coordinate in enumerate(selected_coordinates, start=1):
            offset = np.zeros_like(scales)
            offset[coordinate] = step
            plus = scales + offset
            minus = scales - offset
            if np.min(minus) < 0.0:
                raise ValueError(f"direct hessian step leaves nonnegative scale domain at coordinate {coordinate}")
            if np.max(plus) > args.scale_max:
                raise ValueError(f"direct hessian step exceeds --scale-max at coordinate {coordinate}")
            plus_value, plus_gradient = direct_problem.value_and_grad(plus)
            minus_value, minus_gradient = direct_problem.value_and_grad(minus)
            column = (plus_gradient - minus_gradient) / (2.0 * step)
            hessian_columns[:, column_number - 1] = column
            column_rows.append(
                {
                    "coordinate": int(coordinate),
                    "label": str(group_rows[coordinate]["label"]),
                    "gradient": float(base_gradient[coordinate]),
                    "diagonal_curvature": float(column[coordinate]),
                    "plus_ratio": float(plus_value),
                    "minus_ratio": float(minus_value),
                }
            )
            elapsed = time.perf_counter() - start_time
            per_column = elapsed / column_number
            full_eta = per_column * len(scales)
            print(
                f"direct_hessian_column={column_number}/{len(selected_coordinates)} "
                f"label={group_rows[coordinate]['label']} diag={column[coordinate]:.9e} "
                f"elapsed={elapsed:.1f}s full_eta={full_eta:.1f}s",
                flush=True,
            )

        elapsed = time.perf_counter() - start_time
        per_column = elapsed / max(1, len(selected_coordinates))
        full_estimated_seconds = per_column * len(scales)
        selected_submatrix = hessian_columns[np.ix_(selected_coordinates, range(len(selected_coordinates)))]
        selected_symmetric = 0.5 * (selected_submatrix + selected_submatrix.T)
        selected_eigs = np.linalg.eigvalsh(selected_symmetric)
        full_hessian = None
        full_symmetric = None
        full_eigs = None
        symmetry_max_abs = None
        gradient_direction_curvature = None
        is_full = len(selected_coordinates) == len(scales) and selected_coordinates == list(range(len(scales)))
        if is_full:
            full_hessian = hessian_columns
            symmetry_max_abs = float(np.max(np.abs(full_hessian - full_hessian.T)))
            full_symmetric = 0.5 * (full_hessian + full_hessian.T)
            full_eigs = np.linalg.eigvalsh(full_symmetric)
            grad_norm = float(np.linalg.norm(base_gradient))
            if grad_norm > 0.0:
                grad_unit = base_gradient / grad_norm
                gradient_direction_curvature = float(grad_unit @ full_symmetric @ grad_unit)

        output_summary = {
            "source_json": str(args.direct_hessian_json),
            "backend": args.backend,
            "torch_precision": args.torch_precision,
            "fft_grid": args.fft_grid,
            "step": step,
            "group_count": len(group_rows),
            "active_only": bool(args.direct_hessian_active_only),
            "free_only": bool(args.direct_hessian_free_only),
            "selected_coordinates": selected_coordinates,
            "is_full_hessian": is_full,
            "elapsed_seconds": float(elapsed),
            "seconds_per_column": float(per_column),
            "full_estimated_seconds": float(full_estimated_seconds),
            "base_ratio": float(base_value),
            "gradient_norm": float(np.linalg.norm(base_gradient)),
            "gradient_max_abs": float(np.max(np.abs(base_gradient))),
            "selected_hessian_eigenvalues": [float(value) for value in selected_eigs],
            "selected_hessian_min_eigenvalue": float(selected_eigs[0]),
            "selected_hessian_max_eigenvalue": float(selected_eigs[-1]),
            "full_hessian_symmetry_max_abs": symmetry_max_abs,
            "full_hessian_eigenvalues": None if full_eigs is None else [float(value) for value in full_eigs],
            "full_hessian_min_eigenvalue": None if full_eigs is None else float(full_eigs[0]),
            "full_hessian_max_eigenvalue": None if full_eigs is None else float(full_eigs[-1]),
            "gradient_direction_curvature": gradient_direction_curvature,
            "column_rows": column_rows,
        }
        if full_hessian is not None:
            output_summary["full_hessian"] = [[float(value) for value in row] for row in full_symmetric]
        else:
            output_summary["selected_hessian"] = [[float(value) for value in row] for row in selected_symmetric]

        print(f"direct_hessian_elapsed_seconds={elapsed:.6g}", flush=True)
        print(f"direct_hessian_seconds_per_column={per_column:.6g}", flush=True)
        print(f"direct_hessian_full_estimated_seconds={full_estimated_seconds:.6g}", flush=True)
        print(f"direct_hessian_selected_max_eig={selected_eigs[-1]:.17g}", flush=True)
        if full_eigs is not None:
            print(f"direct_hessian_full_max_eig={full_eigs[-1]:.17g}", flush=True)
            print(f"direct_hessian_full_symmetry_max_abs={symmetry_max_abs:.3e}", flush=True)
            if gradient_direction_curvature is not None:
                print(f"direct_hessian_gradient_direction_curvature={gradient_direction_curvature:.17g}", flush=True)

        if args.direct_hessian_output_json is not None:
            args.direct_hessian_output_json.parent.mkdir(parents=True, exist_ok=True)
            args.direct_hessian_output_json.write_text(json.dumps(output_summary, indent=2) + "\n", encoding="utf-8")
            print(f"direct_hessian_output_json={args.direct_hessian_output_json}", flush=True)
        if args.direct_hessian_md is not None:
            args.direct_hessian_md.parent.mkdir(parents=True, exist_ok=True)
            max_eig_text = (
                f"`{float(full_eigs[-1]):.17g}`" if full_eigs is not None else "`partial benchmark only`"
            )
            lines = [
                "# C* W10/Adaptive Direct Hessian Audit",
                "",
                "This is a tracking artifact, not manuscript text.  It finite-differences",
                "direct CUDA FFT/autograd gradients at the W10/adaptive split branch.",
                "",
                "## Summary",
                "",
                f"- source JSON: `{args.direct_hessian_json}`",
                f"- backend: `{args.backend}`",
                f"- precision: `{args.torch_precision}`",
                f"- fft grid: `{args.fft_grid}`",
                f"- step: `{step:.17g}`",
                f"- active only: `{bool(args.direct_hessian_active_only)}`",
                f"- columns computed: `{len(selected_coordinates)}/{len(scales)}`",
                f"- elapsed seconds: `{elapsed:.6g}`",
                f"- seconds per column: `{per_column:.6g}`",
                f"- full-matrix ETA from this run: `{full_estimated_seconds:.6g}`",
                f"- base ratio: `{base_value:.17g}`",
                f"- gradient norm: `{np.linalg.norm(base_gradient):.17g}`",
                f"- selected max Hessian eigenvalue: `{selected_eigs[-1]:.17g}`",
                f"- full max Hessian eigenvalue: {max_eig_text}",
            ]
            if symmetry_max_abs is not None:
                lines.append(f"- full Hessian symmetry max abs: `{symmetry_max_abs:.3e}`")
            if gradient_direction_curvature is not None:
                lines.append(f"- gradient-direction curvature from Hessian: `{gradient_direction_curvature:.17g}`")
            lines.extend(
                [
                    "",
                    "## Columns",
                    "",
                    "| rank | group | gradient | diagonal curvature | plus ratio | minus ratio |",
                    "|---:|---|---:|---:|---:|---:|",
                ]
            )
            for rank, row in enumerate(column_rows, start=1):
                lines.append(
                    f"| `{rank}` "
                    f"| `{row['label']}` "
                    f"| `{float(row['gradient']):.17g}` "
                    f"| `{float(row['diagonal_curvature']):.17g}` "
                    f"| `{float(row['plus_ratio']):.17g}` "
                    f"| `{float(row['minus_ratio']):.17g}` |"
                )
            lines.extend(
                [
                    "",
                    "## Interpretation",
                    "",
                    "A partial run is an ETA and sign diagnostic only.  A full run gives the",
                    "floating active-face Hessian needed before intervalizing the Schur/KKT",
                    "certificate.  This is still not an interval proof by itself.",
                    "",
                ]
            )
            args.direct_hessian_md.write_text("\n".join(lines), encoding="utf-8")
            print(f"direct_hessian_md={args.direct_hessian_md}", flush=True)
        return

    if args.direct_audit_json is not None:
        if args.backend != "torch-cuda":
            raise ValueError("--direct-audit-json currently requires --backend torch-cuda")
        audit_summary = json.loads(args.direct_audit_json.read_text(encoding="utf-8"))
        if "scales" not in audit_summary or "group_rows" not in audit_summary:
            raise ValueError("--direct-audit-json requires a direct optimize JSON with scales/group_rows")
        scales = np.asarray(audit_summary["scales"], dtype=np.float64)
        group_rows = list(audit_summary["group_rows"])
        if len(scales) != len(group_rows):
            raise ValueError("direct audit scales/group_rows length mismatch")
        audit_steps = [float(item.strip()) for item in args.direct_audit_steps.split(",") if item.strip()]
        if not audit_steps:
            raise ValueError("--direct-audit-steps must contain at least one positive step")
        if any(step <= 0.0 for step in audit_steps):
            raise ValueError("--direct-audit-steps must be positive")

        selected_indices: list[int] = []
        group_ids: list[int] = []
        direction_values: list[np.ndarray] = []
        group_delta_x: list[float] = []
        group_delta_d: list[float] = []
        covered_common: set[int] = set()

        for group_index, row in enumerate(group_rows):
            rank_start = int(row["rank_start"])
            rank_count = int(row["rank_count"])
            rank_end = int(row["rank_end"])
            group_shells, _, _, _, _ = selected_shells_from_optimizer(
                args.csv_path, args.base_s2, rank_start, rank_count
            )
            group_pos, _ = modes_on_shells(set(group_shells), max(group_shells))
            cache_parts = cache_parts_from_group_row(row, args.results_dir)
            direction, part_x2, part_d2, _, group_dx, group_dd, _ = load_direction_cache_parts(
                cache_parts, group_shells, group_pos
            )
            if abs(x2 - part_x2) > 1e-6 or abs(d2 - part_d2) > 1e-4:
                raise ValueError(f"direct audit cache invariant mismatch for {row['label']}")
            for local_index, mode in enumerate(group_pos):
                common_i = common_index[mode]
                if common_i in covered_common:
                    raise ValueError(f"mode {mode} appears in multiple direct audit groups")
                covered_common.add(common_i)
                selected_indices.append(len(active_pos) + common_i)
                group_ids.append(group_index)
                direction_values.append(direction[local_index].copy())
            group_delta_x.append(float(group_dx))
            group_delta_d.append(float(group_dd))

        allow_missing_bound_modes = bool(audit_summary.get("removed_bound_coordinates"))
        if len(covered_common) != len(common_selected_pos) and not allow_missing_bound_modes:
            missing_count = len(common_selected_pos) - len(covered_common)
            raise ValueError(f"direct audit groups do not cover all selected modes; missing {missing_count}")

        direct_problem = TorchCudaDirectScaleProblem(
            all_pos=all_pos,
            base_coeffs_real=active_padded,
            selected_indices=np.asarray(selected_indices, dtype=np.int64),
            group_ids=np.asarray(group_ids, dtype=np.int64),
            direction_values=np.asarray(direction_values, dtype=np.float64),
            e1=e1,
            e2=e2,
            delta_x=np.asarray(group_delta_x, dtype=np.float64),
            delta_d=np.asarray(group_delta_d, dtype=np.float64),
            x2=x2,
            d2=d2,
            grid_size=args.fft_grid,
            precision=args.torch_precision,
        )

        base_value, gradient = direct_problem.value_and_grad(scales)
        grad_norm = float(np.linalg.norm(gradient))
        grad_max = float(np.max(np.abs(gradient)))
        ranked_coordinates = np.argsort(np.abs(gradient))[::-1]

        directions: list[tuple[str, np.ndarray, float | None]] = []
        if grad_norm > 0.0:
            directions.append(("gradient_l2", gradient / grad_norm, None))
        for index in ranked_coordinates[: max(0, args.direct_audit_top_coordinates)]:
            direction = np.zeros_like(scales)
            direction[int(index)] = 1.0
            directions.append((str(group_rows[int(index)]["label"]), direction, float(gradient[int(index)])))

        rows: list[dict[str, object]] = []
        for label, direction, coordinate_gradient in directions:
            for step in audit_steps:
                plus = scales + step * direction
                minus = scales - step * direction
                if np.min(plus) < 0.0 or np.min(minus) < 0.0:
                    print(
                        f"direct_audit_direction={label} step={step:.6g} skipped=nonnegative-bound",
                        flush=True,
                    )
                    continue
                if np.max(plus) > args.scale_max or np.max(minus) > args.scale_max:
                    print(
                        f"direct_audit_direction={label} step={step:.6g} skipped=scale-max",
                        flush=True,
                    )
                    continue
                plus_value = direct_problem.value(plus)
                minus_value = direct_problem.value(minus)
                slope = (plus_value - minus_value) / (2.0 * step)
                curvature = (plus_value - 2.0 * base_value + minus_value) / (step * step)
                row = {
                    "direction": label,
                    "step": step,
                    "slope": float(slope),
                    "curvature": float(curvature),
                    "plus_ratio": float(plus_value),
                    "minus_ratio": float(minus_value),
                    "coordinate_gradient": coordinate_gradient,
                }
                rows.append(row)
                print(
                    f"direct_audit_direction={label} step={step:.6g} slope={slope:.17g} "
                    f"curvature={curvature:.17g} plus={plus_value:.17g} minus={minus_value:.17g}",
                    flush=True,
                )

        print(f"direct_audit_base_ratio={base_value:.17g}", flush=True)
        print(f"direct_audit_gradient_norm={grad_norm:.17g}", flush=True)
        print(f"direct_audit_gradient_max_abs={grad_max:.17g}", flush=True)
        print(
            "direct_audit_top_gradient_components="
            + " ".join(
                f"{group_rows[int(index)]['label']}:{gradient[int(index)]:.6e}"
                for index in ranked_coordinates[: min(12, len(ranked_coordinates))]
            ),
            flush=True,
        )

        gradient_rows = [
            {
                "index": int(index),
                "label": str(group_rows[index]["label"]),
                "rank_start": int(group_rows[index]["rank_start"]),
                "rank_end": int(group_rows[index]["rank_end"]),
                "scale": float(scales[index]),
                "gradient": float(gradient[index]),
                "at_lower_bound": bool(scales[index] <= 1e-12),
            }
            for index in range(len(group_rows))
        ]
        lower_bound_gradients = [row["gradient"] for row in gradient_rows if row["at_lower_bound"]]
        active_gradients = [row["gradient"] for row in gradient_rows if not row["at_lower_bound"]]
        if args.direct_audit_output_json is not None:
            args.direct_audit_output_json.parent.mkdir(parents=True, exist_ok=True)
            audit_output = {
                "source_json": str(args.direct_audit_json),
                "backend": args.backend,
                "torch_precision": args.torch_precision,
                "fft_grid": args.fft_grid,
                "scale_max": float(args.scale_max),
                "base_ratio": float(base_value),
                "gradient_norm": grad_norm,
                "gradient_max_abs": grad_max,
                "active_gradient_max_abs": None
                if not active_gradients
                else float(max(abs(value) for value in active_gradients)),
                "lower_bound_count": len(lower_bound_gradients),
                "lower_bound_gradient_max": None
                if not lower_bound_gradients
                else float(max(lower_bound_gradients)),
                "lower_bound_gradient_min": None
                if not lower_bound_gradients
                else float(min(lower_bound_gradients)),
                "lower_bound_positive_count": int(sum(value > 0.0 for value in lower_bound_gradients)),
                "gradient_rows": gradient_rows,
                "line_rows": rows,
            }
            args.direct_audit_output_json.write_text(json.dumps(audit_output, indent=2) + "\n", encoding="utf-8")
            print(f"direct_audit_output_json={args.direct_audit_output_json}", flush=True)

        if args.direct_audit_md is not None:
            args.direct_audit_md.parent.mkdir(parents=True, exist_ok=True)
            lines = [
                "# C* W10/Adaptive Direct Branch Audit",
                "",
                "This is a tracking artifact, not manuscript text.  It audits direct",
                "alias-safe FFT stationarity and line curvatures for the saved",
                "W10/adaptive split-scale branch.",
                "",
                "## Summary",
                "",
                f"- source JSON: `{args.direct_audit_json}`",
                f"- backend: `{args.backend}`",
                f"- precision: `{args.torch_precision}`",
                f"- fft grid: `{args.fft_grid}`",
                f"- group count: `{len(group_rows)}`",
                f"- base ratio: `{base_value:.17g}`",
                f"- gradient norm: `{grad_norm:.17g}`",
                f"- max absolute gradient component: `{grad_max:.17g}`",
                "",
                "## Top Gradient Components",
                "",
                "| rank | group | gradient | scale |",
                "|---:|---|---:|---:|",
            ]
            top_gradient_count = min(max(20, args.direct_audit_top_coordinates), len(ranked_coordinates))
            for rank, index in enumerate(ranked_coordinates[:top_gradient_count], start=1):
                lines.append(
                    f"| `{rank}` "
                    f"| `{group_rows[int(index)]['label']}` "
                    f"| `{gradient[int(index)]:.17g}` "
                    f"| `{scales[int(index)]:.17g}` |"
                )
            lines.extend(
                [
                    "",
                    "## Line Curvatures",
                    "",
                    "| direction | step | slope | curvature | plus ratio | minus ratio | coordinate gradient |",
                    "|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in rows:
                coordinate_gradient = row["coordinate_gradient"]
                coordinate_text = "" if coordinate_gradient is None else f"{float(coordinate_gradient):.17g}"
                lines.append(
                    f"| `{row['direction']}` "
                    f"| `{float(row['step']):.6g}` "
                    f"| `{float(row['slope']):.17g}` "
                    f"| `{float(row['curvature']):.17g}` "
                    f"| `{float(row['plus_ratio']):.17g}` "
                    f"| `{float(row['minus_ratio']):.17g}` "
                    f"| `{coordinate_text}` |"
                )
            lines.extend(
                [
                    "",
                    "## Interpretation",
                    "",
                    "This audit checks the current split branch at the direct FFT level.  A",
                    "small but nonzero gradient means the point is a strong lower-bound",
                    "checkpoint, but a final stationarity certificate still needs either a",
                    "tighter local optimization or interval KKT tolerances that explicitly",
                    "cover the residual gradient.",
                    "",
                ]
            )
            args.direct_audit_md.write_text("\n".join(lines), encoding="utf-8")
            print(f"direct_audit_md={args.direct_audit_md}", flush=True)
        return

    if args.benchmark_only:
        rng = np.random.default_rng(args.seed)
        points = [np.full(len(blocks), 1.3470584372295857, dtype=np.float64)]
        points.extend(rng.uniform(0.0, args.sample_scale_max, size=len(blocks)) for _ in range(max(0, args.benchmark_evals - 1)))
        if hasattr(evaluator, "torch"):
            evaluator.torch.cuda.synchronize()
        start_time = time.perf_counter()
        for index, point in enumerate(points, start=1):
            value, imag = evaluator(point)
            print(f"benchmark_eval={index}/{len(points)} value={value:.17g} imag={imag:.3e}", flush=True)
        if hasattr(evaluator, "torch"):
            evaluator.torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        per_eval = elapsed / max(1, len(points))
        print(f"benchmark_backend={args.backend}", flush=True)
        print(f"benchmark_evals={len(points)}", flush=True)
        print(f"benchmark_elapsed_seconds={elapsed:.6g}", flush=True)
        print(f"benchmark_seconds_per_eval={per_eval:.6g}", flush=True)
        return

    dim = len(blocks)
    exponents = monomial_exponents(dim, 3)
    probe_scale = 1.3470584372295857
    sample_points = build_sample_points(
        dim, max(args.fit_samples, len(exponents) + 8), args.sample_scale_max, probe_scale, args.seed, exponents
    )
    print(f"fitting samples={len(sample_points)} coefficients={len(exponents)}", flush=True)
    values = []
    imaginary = []
    for index, point in enumerate(sample_points, start=1):
        value, imag = evaluator(point)
        values.append(value)
        imaginary.append(abs(imag))
        if index == 1 or index % 12 == 0 or index == len(sample_points):
            print(f"fft_fit_progress={index}/{len(sample_points)} value={value:.17g} imag={imag:.3e}", flush=True)
    design = monomial_design(sample_points, exponents)
    coeffs, *_ = np.linalg.lstsq(design, np.asarray(values, dtype=np.float64), rcond=None)
    fitted = design @ coeffs
    fit_residual = np.abs(fitted - np.asarray(values, dtype=np.float64))

    rng = np.random.default_rng(args.seed + 1)
    validation_points = rng.uniform(0.0, args.sample_scale_max, size=(args.validation_samples, dim))
    validation_residuals = []
    for index, point in enumerate(validation_points, start=1):
        actual, imag = evaluator(point)
        predicted = polynomial_value(point, coeffs, exponents)
        validation_residuals.append(abs(actual - predicted))
        print(
            f"fft_validation={index}/{len(validation_points)} actual={actual:.17g} "
            f"predicted={predicted:.17g} residual={abs(actual - predicted):.3e} imag={imag:.3e}",
            flush=True,
        )

    delta_x = np.asarray(block_dx, dtype=np.float64)
    delta_d = np.asarray(block_dd, dtype=np.float64)
    scalar_result = minimize_scalar(
        lambda scale: -scalar_ratio(scale, coeffs, exponents, delta_x, delta_d),
        bounds=(0.0, args.scale_max),
        method="bounded",
        options={"xatol": 1e-12},
    )
    scalar_candidates = [
        (0.0, scalar_ratio(0.0, coeffs, exponents, delta_x, delta_d)),
        (args.scale_max, scalar_ratio(args.scale_max, coeffs, exponents, delta_x, delta_d)),
    ]
    if scalar_result.success:
        scalar_candidates.append((float(scalar_result.x), -float(scalar_result.fun)))
    scalar_best_scale, scalar_best_ratio = max(scalar_candidates, key=lambda item: item[1])

    def objective(z: np.ndarray) -> float:
        return -ratio_from_poly(z, coeffs, exponents, delta_x, delta_d)

    starts = [
        np.full(dim, scalar_best_scale, dtype=np.float64),
        np.ones(dim, dtype=np.float64),
        np.linspace(1.35, 2.15, dim, dtype=np.float64),
        np.linspace(2.15, 1.15, dim, dtype=np.float64),
    ]
    best = None
    for start in starts:
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=[(0.0, args.scale_max)] * dim,
            options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 2000, "maxls": 50},
        )
        value = -float(result.fun)
        if best is None or value > best[0]:
            best = (value, np.asarray(result.x, dtype=np.float64), bool(result.success), int(result.nit))
    assert best is not None
    free_best_ratio, free_scales, free_success, free_iterations = best

    ratio_func = lambda z: ratio_from_poly(np.asarray(z, dtype=np.float64), coeffs, exponents, delta_x, delta_d)
    scalar_point = np.full(dim, scalar_best_scale, dtype=np.float64)
    scalar_gradient = np.zeros(dim, dtype=np.float64)
    step = args.hessian_step
    for coordinate in range(dim):
        offset = np.zeros(dim, dtype=np.float64)
        offset[coordinate] = step
        scalar_gradient[coordinate] = (ratio_func(scalar_point + offset) - ratio_func(scalar_point - offset)) / (2 * step)
    scalar_direction = np.ones(dim, dtype=np.float64)
    scalar_direction /= np.linalg.norm(scalar_direction)
    transverse_gradient = scalar_gradient - scalar_direction * float(scalar_gradient @ scalar_direction)
    hessian = finite_hessian(ratio_func, free_scales, step)
    hessian_eigs = np.linalg.eigvalsh(0.5 * (hessian + hessian.T))

    validation_max = float(max(validation_residuals) if validation_residuals else 0.0)
    fit_max = float(np.max(fit_residual))
    scalar_branch_stationary = bool(np.linalg.norm(transverse_gradient) <= 10.0 * max(validation_max, fit_max, 1e-13))

    summary: dict[str, object] = {
        "fft_grid": args.fft_grid,
        "layout": layout_id,
        "alias_safe": alias_safe,
        "backend": args.backend,
        "fft_workers": int(args.fft_workers),
        "torch_precision": args.torch_precision,
        "max_coord": max_coord,
        "fit_samples": len(sample_points),
        "validation_samples": len(validation_points),
        "fit_residual_max": fit_max,
        "fit_residual_l2": float(np.linalg.norm(fit_residual)),
        "validation_residual_max": validation_max,
        "imaginary_residual_max": float(max(imaginary) if imaginary else 0.0),
        "scalar_best_scale": float(scalar_best_scale),
        "scalar_best_ratio": float(scalar_best_ratio),
        "free_best_ratio": float(free_best_ratio),
        "free_gain_over_scalar": float(free_best_ratio - scalar_best_ratio),
        "free_scales": [float(value) for value in free_scales],
        "free_success": free_success,
        "free_iterations": free_iterations,
        "scalar_gradient": [float(value) for value in scalar_gradient],
        "scalar_transverse_gradient_norm": float(np.linalg.norm(transverse_gradient)),
        "free_hessian_eigenvalues": [float(value) for value in hessian_eigs],
        "free_hessian_max_eigenvalue": float(hessian_eigs[-1]),
        "scalar_branch_stationary": scalar_branch_stationary,
        "block_rows": block_rows,
        "monomial_exponents": [list(item) for item in exponents],
        "polynomial_coefficients": [float(value) for value in coeffs],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.output_md, summary, block_rows)

    print(f"scalar_best_scale={scalar_best_scale:.17g}", flush=True)
    print(f"scalar_best_ratio={scalar_best_ratio:.17g}", flush=True)
    print(f"free_best_ratio={free_best_ratio:.17g}", flush=True)
    print(f"free_gain_over_scalar={free_best_ratio - scalar_best_ratio:.17g}", flush=True)
    print(f"free_scales={' '.join(f'{value:.12g}' for value in free_scales)}", flush=True)
    print(f"scalar_transverse_gradient_norm={np.linalg.norm(transverse_gradient):.17g}", flush=True)
    print(f"free_hessian_eigs={' '.join(f'{value:.9e}' for value in hessian_eigs)}", flush=True)
    print(f"output_md={args.output_md}", flush=True)
    print(f"output_json={args.output_json}", flush=True)


if __name__ == "__main__":
    main()
