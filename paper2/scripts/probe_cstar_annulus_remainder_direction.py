"""Probe nonlinear annulus remainders along the one-high optimizer direction.

This is a targeted diagnostic for the C* annulus closure problem.  It does not
build the full 565->2260 release ledger.  Instead, it selects the shells that
carry the largest one-high linear-envelope contribution, reconstructs the
gradient direction on those shells, and evaluates the exact nonlinear ratio on
that selected shell set.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from analyze_cstar_annulus_linear_optimizer import load_shell_norms, optimize, value_and_grad
from optimize_cstar_release_band import load_base_coeffs
from probe_cstar_inactive_gradient import (
    DEFAULT_COEFF_DIR,
    add_modes,
    append_triad,
    basis_arrays,
    positive_modes,
    shell,
    subtract_modes,
)
from shell_decomp import get_modes


def modes_on_shells(shells: set[int], cap_s2: int) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    selected_pos = [mode for mode in positive_modes(cap_s2) if shell(mode) in shells]
    selected_full = [mode for mode in get_modes(cap_s2) if shell(mode) in shells]
    return selected_pos, selected_full


def build_selected_one_high_triads(
    base_s2: int,
    active_pos: list[tuple[int, int, int]],
    selected_pos: list[tuple[int, int, int]],
    selected_full: list[tuple[int, int, int]],
    pos_index: dict[tuple[int, int, int], int],
) -> dict[str, np.ndarray]:
    active_full = get_modes(base_s2)
    active_full_set = set(active_full)
    active_pos_set = set(active_pos)

    ell_idxs: list[int] = []
    ell2s: list[float] = []
    r_idxs: list[int] = []
    r_conjs: list[bool] = []
    s_idxs: list[int] = []
    s_conjs: list[bool] = []
    s_vecs: list[tuple[float, float, float]] = []

    for ell in selected_pos:
        for r in active_full:
            s = subtract_modes(ell, r)
            if s in active_full_set:
                append_triad(ell, r, s, pos_index, ell_idxs, ell2s, r_idxs, r_conjs, s_idxs, s_conjs, s_vecs)

    for ell in active_pos:
        for r in selected_full:
            s = subtract_modes(ell, r)
            if s in active_full_set:
                append_triad(ell, r, s, pos_index, ell_idxs, ell2s, r_idxs, r_conjs, s_idxs, s_conjs, s_vecs)

    for s in selected_full:
        for r in active_full:
            ell = add_modes(r, s)
            if ell in active_pos_set:
                append_triad(ell, r, s, pos_index, ell_idxs, ell2s, r_idxs, r_conjs, s_idxs, s_conjs, s_vecs)

    return {
        "ell_idxs": np.array(ell_idxs, dtype=np.int64),
        "ell2s": np.array(ell2s, dtype=np.float64),
        "r_idxs": np.array(r_idxs, dtype=np.int64),
        "r_conjs": np.array(r_conjs, dtype=bool),
        "s_idxs": np.array(s_idxs, dtype=np.int64),
        "s_conjs": np.array(s_conjs, dtype=bool),
        "s_vecs": np.array(s_vecs, dtype=np.float64),
    }


def compute_selected_inactive_gradient_streaming(
    base_s2: int,
    active_pos: list[tuple[int, int, int]],
    selected_pos: list[tuple[int, int, int]],
    selected_full: list[tuple[int, int, int]],
    pos_index: dict[tuple[int, int, int], int],
    active_coeffs_np: np.ndarray,
    e1_np: np.ndarray,
    e2_np: np.ndarray,
    k2_np: np.ndarray,
    device: torch.device,
    batch_triads: int,
    progress_triads: int,
    worker_count: int = 1,
) -> tuple[float, float, np.ndarray, int]:
    if worker_count > 1:
        return compute_selected_inactive_gradient_streaming_parallel(
            base_s2,
            active_pos,
            selected_pos,
            selected_full,
            pos_index,
            active_coeffs_np,
            e1_np,
            e2_np,
            k2_np,
            str(device),
            batch_triads,
            progress_triads,
            worker_count,
        )
    return compute_selected_inactive_gradient_streaming_shard(
        base_s2,
        active_pos,
        selected_pos,
        selected_full,
        pos_index,
        active_coeffs_np,
        e1_np,
        e2_np,
        k2_np,
        str(device),
        batch_triads,
        progress_triads,
        0,
        1,
    )


def compute_selected_inactive_gradient_streaming_parallel(
    base_s2: int,
    active_pos: list[tuple[int, int, int]],
    selected_pos: list[tuple[int, int, int]],
    selected_full: list[tuple[int, int, int]],
    pos_index: dict[tuple[int, int, int], int],
    active_coeffs_np: np.ndarray,
    e1_np: np.ndarray,
    e2_np: np.ndarray,
    k2_np: np.ndarray,
    device_name: str,
    batch_triads: int,
    progress_triads: int,
    worker_count: int,
) -> tuple[float, float, np.ndarray, int]:
    print(f"one_high_workers={worker_count}", flush=True)
    gradient_total = np.zeros((len(selected_pos), 4), dtype=np.float64)
    total_count = 0
    x2_value = d2_value = None
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                compute_selected_inactive_gradient_streaming_shard,
                base_s2,
                active_pos,
                selected_pos,
                selected_full,
                pos_index,
                active_coeffs_np,
                e1_np,
                e2_np,
                k2_np,
                device_name,
                batch_triads,
                progress_triads,
                worker_index,
                worker_count,
            )
            for worker_index in range(worker_count)
        ]
        for future in as_completed(futures):
            x2, d2, gradient, count = future.result()
            if x2_value is None:
                x2_value = x2
                d2_value = d2
            elif abs(x2_value - x2) > 1e-6 or abs(d2_value - d2) > 1e-4:
                raise ValueError("parallel one-high worker invariant mismatch")
            gradient_total += gradient
            total_count += count
    if x2_value is None or d2_value is None:
        raise ValueError("no one-high worker results")
    return x2_value, d2_value, gradient_total, total_count


def compute_selected_inactive_gradient_streaming_shard(
    base_s2: int,
    active_pos: list[tuple[int, int, int]],
    selected_pos: list[tuple[int, int, int]],
    selected_full: list[tuple[int, int, int]],
    pos_index: dict[tuple[int, int, int], int],
    active_coeffs_np: np.ndarray,
    e1_np: np.ndarray,
    e2_np: np.ndarray,
    k2_np: np.ndarray,
    device_name: str,
    batch_triads: int,
    progress_triads: int,
    worker_index: int,
    worker_count: int,
) -> tuple[float, float, np.ndarray, int]:
    active_full = get_modes(base_s2)
    active_full_set = set(active_full)
    active_pos_set = set(active_pos)
    selected_pos_shard = selected_pos[worker_index::worker_count]
    selected_full_shard = selected_full[worker_index::worker_count]

    device = torch.device(device_name)
    dtype = torch.float64
    use_cuda = device.type == "cuda"
    active_coeffs = torch.tensor(active_coeffs_np, dtype=dtype, device=device)
    inactive_coeffs = torch.zeros((len(selected_pos), 4), dtype=dtype, device=device, requires_grad=True)
    e1 = torch.tensor(e1_np, dtype=dtype, device=device)
    e2 = torch.tensor(e2_np, dtype=dtype, device=device)
    k2 = torch.tensor(k2_np, dtype=dtype, device=device)
    k4 = k2 * k2
    zeros3 = torch.zeros_like(e1)
    e1_c = torch.complex(e1, zeros3)
    e2_c = torch.complex(e2, zeros3)

    coeffs = torch.cat([active_coeffs, inactive_coeffs], dim=0)
    c1 = torch.view_as_complex(coeffs[:, :2].contiguous())
    c2 = torch.view_as_complex(coeffs[:, 2:].contiguous())
    u = c1.unsqueeze(1) * e1_c + c2.unsqueeze(1) * e2_c

    amp2 = 2.0 * u.abs().pow(2).sum(dim=1)
    x2 = (k2 * amp2).sum()
    d2 = (k4 * amp2).sum()
    denominator = x2.detach() * d2.detach().sqrt()
    u_leaf = u.detach().clone().requires_grad_(True)

    ell_idxs: list[int] = []
    ell2s: list[float] = []
    r_idxs: list[int] = []
    r_conjs: list[bool] = []
    s_idxs: list[int] = []
    s_conjs: list[bool] = []
    s_vecs: list[tuple[float, float, float]] = []

    total_count = 0
    next_progress = progress_triads

    def flush_batch() -> None:
        if not ell_idxs:
            return
        ell_i = torch.tensor(ell_idxs, dtype=torch.long)
        ell2 = torch.tensor(ell2s, dtype=dtype)
        r_i = torch.tensor(r_idxs, dtype=torch.long)
        r_conj = torch.tensor(r_conjs, dtype=torch.bool)
        s_i = torch.tensor(s_idxs, dtype=torch.long)
        s_conj = torch.tensor(s_conjs, dtype=torch.bool)
        s_vecs_real = torch.tensor(s_vecs, dtype=dtype)
        s_vec = torch.complex(s_vecs_real, torch.zeros_like(s_vecs_real))
        if use_cuda:
            ell_i = ell_i.pin_memory()
            ell2 = ell2.pin_memory()
            r_i = r_i.pin_memory()
            r_conj = r_conj.pin_memory()
            s_i = s_i.pin_memory()
            s_conj = s_conj.pin_memory()
            s_vec = s_vec.pin_memory()

        ell_i = ell_i.to(device, non_blocking=True)
        r_i = r_i.to(device, non_blocking=True)
        s_i = s_i.to(device, non_blocking=True)
        r_conj = r_conj.to(device, non_blocking=True)
        s_conj = s_conj.to(device, non_blocking=True)
        ell2 = ell2.to(device, non_blocking=True)
        s_vec = s_vec.to(device, non_blocking=True)
        u_ell = u_leaf[ell_i]
        u_r = u_leaf[r_i]
        u_s = u_leaf[s_i]
        u_r = torch.where(r_conj.unsqueeze(1), u_r.conj(), u_r)
        u_s = torch.where(s_conj.unsqueeze(1), u_s.conj(), u_s)
        s_dot_ur = (s_vec * u_r).sum(dim=1)
        uell_dot_us = (u_ell.conj() * u_s).sum(dim=1)
        b_chunk = 2.0 * (-ell2 * (s_dot_ur * uell_dot_us).imag).sum()
        b_chunk.backward()

        ell_idxs.clear()
        ell2s.clear()
        r_idxs.clear()
        r_conjs.clear()
        s_idxs.clear()
        s_conjs.clear()
        s_vecs.clear()

    def append_streamed(ell: tuple[int, int, int], r: tuple[int, int, int], s: tuple[int, int, int]) -> None:
        nonlocal total_count, next_progress
        append_triad(ell, r, s, pos_index, ell_idxs, ell2s, r_idxs, r_conjs, s_idxs, s_conjs, s_vecs)
        total_count += 1
        if len(ell_idxs) >= batch_triads:
            flush_batch()
        if progress_triads > 0 and total_count >= next_progress:
            if worker_count == 1:
                print(f"streamed_one_high_triads={total_count}", flush=True)
            else:
                print(
                    f"one_high_worker={worker_index + 1}/{worker_count} streamed_one_high_triads={total_count}",
                    flush=True,
                )
            next_progress += progress_triads

    for ell in selected_pos_shard:
        for r in active_full:
            s = subtract_modes(ell, r)
            if s in active_full_set:
                append_streamed(ell, r, s)

    for ell in active_pos:
        for r in selected_full_shard:
            s = subtract_modes(ell, r)
            if s in active_full_set:
                append_streamed(ell, r, s)

    for s in selected_full_shard:
        for r in active_full:
            ell = add_modes(r, s)
            if ell in active_pos_set:
                append_streamed(ell, r, s)

    flush_batch()
    u.backward(gradient=u_leaf.grad)
    gradient = inactive_coeffs.grad.detach() / denominator
    return float(x2.cpu()), float(d2.cpu()), gradient.cpu().numpy(), total_count


def build_selected_release_triads(
    base_s2: int,
    active_pos: list[tuple[int, int, int]],
    selected_pos: list[tuple[int, int, int]],
    selected_full: list[tuple[int, int, int]],
    pos_index: dict[tuple[int, int, int], int],
) -> dict[str, np.ndarray]:
    active_full = get_modes(base_s2)
    active_full_set = set(active_full)
    active_pos_set = set(active_pos)
    selected_full_set = set(selected_full)
    all_full = active_full + selected_full
    all_full_set = active_full_set | selected_full_set

    ell_idxs: list[int] = []
    ell2s: list[float] = []
    r_idxs: list[int] = []
    r_conjs: list[bool] = []
    s_idxs: list[int] = []
    s_conjs: list[bool] = []
    s_vecs: list[tuple[float, float, float]] = []

    for ell in active_pos:
        for s in selected_full:
            r = subtract_modes(ell, s)
            if r in active_full_set:
                append_triad(ell, r, s, pos_index, ell_idxs, ell2s, r_idxs, r_conjs, s_idxs, s_conjs, s_vecs)

    for ell in active_pos:
        for r in selected_full:
            s = subtract_modes(ell, r)
            if s in all_full_set:
                append_triad(ell, r, s, pos_index, ell_idxs, ell2s, r_idxs, r_conjs, s_idxs, s_conjs, s_vecs)

    for ell in selected_pos:
        for r in all_full:
            s = subtract_modes(ell, r)
            if s in all_full_set:
                append_triad(ell, r, s, pos_index, ell_idxs, ell2s, r_idxs, r_conjs, s_idxs, s_conjs, s_vecs)

    return {
        "ell_idxs": np.array(ell_idxs, dtype=np.int64),
        "ell2s": np.array(ell2s, dtype=np.float64),
        "r_idxs": np.array(r_idxs, dtype=np.int64),
        "r_conjs": np.array(r_conjs, dtype=bool),
        "s_idxs": np.array(s_idxs, dtype=np.int64),
        "s_conjs": np.array(s_conjs, dtype=bool),
        "s_vecs": np.array(s_vecs, dtype=np.float64),
    }


def eval_selected_release_line_streaming(
    base_s2: int,
    active_pos: list[tuple[int, int, int]],
    selected_pos: list[tuple[int, int, int]],
    selected_full: list[tuple[int, int, int]],
    pos_index: dict[tuple[int, int, int], int],
    active_coeffs_np: np.ndarray,
    inactive_coeffs_np: np.ndarray,
    e1_np: np.ndarray,
    e2_np: np.ndarray,
    k2_np: np.ndarray,
    base_b: float,
    device: torch.device,
    batch_triads: int,
    progress_triads: int,
) -> tuple[float, int]:
    active_full = get_modes(base_s2)
    active_full_set = set(active_full)
    selected_full_set = set(selected_full)
    all_full = active_full + selected_full
    all_full_set = active_full_set | selected_full_set

    dtype = torch.float64
    use_cuda = device.type == "cuda"
    coeffs_np = np.vstack([active_coeffs_np, inactive_coeffs_np])
    coeffs = torch.tensor(coeffs_np, dtype=dtype, device=device)
    e1 = torch.tensor(e1_np, dtype=dtype, device=device)
    e2 = torch.tensor(e2_np, dtype=dtype, device=device)
    k2 = torch.tensor(k2_np, dtype=dtype, device=device)
    k4 = k2 * k2
    zeros3 = torch.zeros_like(e1)
    e1_c = torch.complex(e1, zeros3)
    e2_c = torch.complex(e2, zeros3)

    c1 = torch.view_as_complex(coeffs[:, :2].contiguous())
    c2 = torch.view_as_complex(coeffs[:, 2:].contiguous())
    u = c1.unsqueeze(1) * e1_c + c2.unsqueeze(1) * e2_c
    amp2 = 2.0 * u.abs().pow(2).sum(dim=1)
    x2 = (k2 * amp2).sum()
    d2 = (k4 * amp2).sum()

    ell_idxs: list[int] = []
    ell2s: list[float] = []
    r_idxs: list[int] = []
    r_conjs: list[bool] = []
    s_idxs: list[int] = []
    s_conjs: list[bool] = []
    s_vecs: list[tuple[float, float, float]] = []

    b_delta = torch.zeros((), dtype=dtype, device=device)
    total_count = 0
    next_progress = progress_triads

    def flush_batch() -> None:
        nonlocal b_delta
        if not ell_idxs:
            return
        ell_i = torch.tensor(ell_idxs, dtype=torch.long)
        ell2 = torch.tensor(ell2s, dtype=dtype)
        r_i = torch.tensor(r_idxs, dtype=torch.long)
        r_conj = torch.tensor(r_conjs, dtype=torch.bool)
        s_i = torch.tensor(s_idxs, dtype=torch.long)
        s_conj = torch.tensor(s_conjs, dtype=torch.bool)
        s_vecs_real = torch.tensor(s_vecs, dtype=dtype)
        s_vec = torch.complex(s_vecs_real, torch.zeros_like(s_vecs_real))
        if use_cuda:
            ell_i = ell_i.pin_memory()
            ell2 = ell2.pin_memory()
            r_i = r_i.pin_memory()
            r_conj = r_conj.pin_memory()
            s_i = s_i.pin_memory()
            s_conj = s_conj.pin_memory()
            s_vec = s_vec.pin_memory()

        with torch.no_grad():
            ell_i = ell_i.to(device, non_blocking=True)
            r_i = r_i.to(device, non_blocking=True)
            s_i = s_i.to(device, non_blocking=True)
            r_conj = r_conj.to(device, non_blocking=True)
            s_conj = s_conj.to(device, non_blocking=True)
            ell2 = ell2.to(device, non_blocking=True)
            s_vec = s_vec.to(device, non_blocking=True)
            u_ell = u[ell_i]
            u_r = u[r_i]
            u_s = u[s_i]
            u_r = torch.where(r_conj.unsqueeze(1), u_r.conj(), u_r)
            u_s = torch.where(s_conj.unsqueeze(1), u_s.conj(), u_s)
            s_dot_ur = (s_vec * u_r).sum(dim=1)
            uell_dot_us = (u_ell.conj() * u_s).sum(dim=1)
            b_delta += 2.0 * (-ell2 * (s_dot_ur * uell_dot_us).imag).sum()

        ell_idxs.clear()
        ell2s.clear()
        r_idxs.clear()
        r_conjs.clear()
        s_idxs.clear()
        s_conjs.clear()
        s_vecs.clear()

    def append_streamed(ell: tuple[int, int, int], r: tuple[int, int, int], s: tuple[int, int, int]) -> None:
        nonlocal total_count, next_progress
        append_triad(ell, r, s, pos_index, ell_idxs, ell2s, r_idxs, r_conjs, s_idxs, s_conjs, s_vecs)
        total_count += 1
        if len(ell_idxs) >= batch_triads:
            flush_batch()
        if progress_triads > 0 and total_count >= next_progress:
            print(f"streamed_release_triads={total_count}", flush=True)
            next_progress += progress_triads

    for ell in active_pos:
        for s in selected_full:
            r = subtract_modes(ell, s)
            if r in active_full_set:
                append_streamed(ell, r, s)

    for ell in active_pos:
        for r in selected_full:
            s = subtract_modes(ell, r)
            if s in all_full_set:
                append_streamed(ell, r, s)

    for ell in selected_pos:
        for r in all_full:
            s = subtract_modes(ell, r)
            if s in all_full_set:
                append_streamed(ell, r, s)

    flush_batch()
    ratio = (torch.tensor(base_b, dtype=dtype, device=device) + b_delta) / (x2 * d2.sqrt())
    return float(ratio.cpu()), total_count


def eval_selected_release_polynomial_streaming(
    base_s2: int,
    active_pos: list[tuple[int, int, int]],
    selected_shells: list[int],
    selected_pos: list[tuple[int, int, int]],
    selected_full: list[tuple[int, int, int]],
    pos_index: dict[tuple[int, int, int], int],
    active_coeffs_np: np.ndarray,
    inactive_direction_np: np.ndarray,
    e1_np: np.ndarray,
    e2_np: np.ndarray,
    k2_np: np.ndarray,
    device: torch.device,
    batch_triads: int,
    progress_triads: int,
    worker_count: int = 1,
) -> tuple[np.ndarray, np.ndarray, int]:
    if worker_count > 1:
        return eval_selected_release_polynomial_streaming_parallel(
            base_s2,
            active_pos,
            selected_shells,
            selected_pos,
            selected_full,
            pos_index,
            active_coeffs_np,
            inactive_direction_np,
            e1_np,
            e2_np,
            k2_np,
            str(device),
            batch_triads,
            progress_triads,
            worker_count,
        )
    return eval_selected_release_polynomial_streaming_shard(
        base_s2,
        active_pos,
        selected_shells,
        selected_pos,
        selected_full,
        pos_index,
        active_coeffs_np,
        inactive_direction_np,
        e1_np,
        e2_np,
        k2_np,
        str(device),
        batch_triads,
        progress_triads,
        0,
        1,
    )


def eval_selected_release_polynomial_streaming_parallel(
    base_s2: int,
    active_pos: list[tuple[int, int, int]],
    selected_shells: list[int],
    selected_pos: list[tuple[int, int, int]],
    selected_full: list[tuple[int, int, int]],
    pos_index: dict[tuple[int, int, int], int],
    active_coeffs_np: np.ndarray,
    inactive_direction_np: np.ndarray,
    e1_np: np.ndarray,
    e2_np: np.ndarray,
    k2_np: np.ndarray,
    device_name: str,
    batch_triads: int,
    progress_triads: int,
    worker_count: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    print(f"release_workers={worker_count}", flush=True)
    numerator_total = np.zeros(3, dtype=np.float64)
    matrix_total = np.zeros((len(selected_shells), len(selected_shells)), dtype=np.float64)
    triad_count_total = 0
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                eval_selected_release_polynomial_streaming_shard,
                base_s2,
                active_pos,
                selected_shells,
                selected_pos,
                selected_full,
                pos_index,
                active_coeffs_np,
                inactive_direction_np,
                e1_np,
                e2_np,
                k2_np,
                device_name,
                batch_triads,
                progress_triads,
                worker_index,
                worker_count,
            )
            for worker_index in range(worker_count)
        ]
        for future in as_completed(futures):
            numerator_coeffs, quadratic_matrix, triad_count = future.result()
            numerator_total += numerator_coeffs
            matrix_total += quadratic_matrix
            triad_count_total += triad_count
    return numerator_total, matrix_total, triad_count_total


def eval_selected_release_polynomial_streaming_shard(
    base_s2: int,
    active_pos: list[tuple[int, int, int]],
    selected_shells: list[int],
    selected_pos: list[tuple[int, int, int]],
    selected_full: list[tuple[int, int, int]],
    pos_index: dict[tuple[int, int, int], int],
    active_coeffs_np: np.ndarray,
    inactive_direction_np: np.ndarray,
    e1_np: np.ndarray,
    e2_np: np.ndarray,
    k2_np: np.ndarray,
    device_name: str,
    batch_triads: int,
    progress_triads: int,
    worker_index: int,
    worker_count: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if worker_count < 1:
        raise ValueError("worker_count must be >= 1")
    if worker_index < 0 or worker_index >= worker_count:
        raise ValueError("worker_index must satisfy 0 <= worker_index < worker_count")

    device = torch.device(device_name)
    active_full = get_modes(base_s2)
    active_full_set = set(active_full)
    selected_full_set = set(selected_full)
    all_full = active_full + selected_full
    all_full_set = active_full_set | selected_full_set
    shell_to_matrix_index = {shell_value: index for index, shell_value in enumerate(selected_shells)}
    group_lookup_np = np.full(len(active_pos) + len(selected_pos), -1, dtype=np.int64)
    for index, mode in enumerate(selected_pos, start=len(active_pos)):
        group_lookup_np[index] = shell_to_matrix_index[shell(mode)]

    dtype = torch.float64
    use_cuda = device.type == "cuda"
    active_zero = np.zeros_like(active_coeffs_np)
    inactive_zero = np.zeros_like(inactive_direction_np)
    coeffs_active_np = np.vstack([active_coeffs_np, inactive_zero])
    coeffs_direction_np = np.vstack([active_zero, inactive_direction_np])

    e1 = torch.tensor(e1_np, dtype=dtype, device=device)
    e2 = torch.tensor(e2_np, dtype=dtype, device=device)
    k2 = torch.tensor(k2_np, dtype=dtype, device=device)
    k4 = k2 * k2
    zeros3 = torch.zeros_like(e1)
    e1_c = torch.complex(e1, zeros3)
    e2_c = torch.complex(e2, zeros3)

    def velocity(coeffs_np: np.ndarray) -> torch.Tensor:
        coeffs = torch.tensor(coeffs_np, dtype=dtype, device=device)
        c1 = torch.view_as_complex(coeffs[:, :2].contiguous())
        c2 = torch.view_as_complex(coeffs[:, 2:].contiguous())
        return c1.unsqueeze(1) * e1_c + c2.unsqueeze(1) * e2_c

    u0 = velocity(coeffs_active_np)
    u1 = velocity(coeffs_direction_np)
    amp2 = 2.0 * u0.abs().pow(2).sum(dim=1)
    x2 = (k2 * amp2).sum()
    d2 = (k4 * amp2).sum()
    denominator = x2 * d2.sqrt()

    ell_idxs: list[int] = []
    ell2s: list[float] = []
    r_idxs: list[int] = []
    r_conjs: list[bool] = []
    s_idxs: list[int] = []
    s_conjs: list[bool] = []
    s_vecs: list[tuple[float, float, float]] = []

    b_coeffs = torch.zeros(3, dtype=dtype, device=device)
    quadratic_matrix = torch.zeros((len(selected_shells), len(selected_shells)), dtype=dtype, device=device)
    group_lookup = torch.tensor(group_lookup_np, dtype=torch.long, device=device)
    total_count = 0
    next_progress = progress_triads

    def flush_batch() -> None:
        nonlocal b_coeffs
        if not ell_idxs:
            return
        ell_i = torch.tensor(ell_idxs, dtype=torch.long)
        ell2 = torch.tensor(ell2s, dtype=dtype)
        r_i = torch.tensor(r_idxs, dtype=torch.long)
        r_conj = torch.tensor(r_conjs, dtype=torch.bool)
        s_i = torch.tensor(s_idxs, dtype=torch.long)
        s_conj = torch.tensor(s_conjs, dtype=torch.bool)
        s_vecs_real = torch.tensor(s_vecs, dtype=dtype)
        s_vec = torch.complex(s_vecs_real, torch.zeros_like(s_vecs_real))
        if use_cuda:
            ell_i = ell_i.pin_memory()
            ell2 = ell2.pin_memory()
            r_i = r_i.pin_memory()
            r_conj = r_conj.pin_memory()
            s_i = s_i.pin_memory()
            s_conj = s_conj.pin_memory()
            s_vec = s_vec.pin_memory()

        with torch.no_grad():
            ell_i = ell_i.to(device, non_blocking=True)
            r_i = r_i.to(device, non_blocking=True)
            s_i = s_i.to(device, non_blocking=True)
            r_conj = r_conj.to(device, non_blocking=True)
            s_conj = s_conj.to(device, non_blocking=True)
            ell2 = ell2.to(device, non_blocking=True)
            s_vec = s_vec.to(device, non_blocking=True)
            ell_group = group_lookup[ell_i]
            r_group = group_lookup[r_i]
            s_group = group_lookup[s_i]

            u0_ell = u0[ell_i]
            u1_ell = u1[ell_i]
            u0_r = u0[r_i]
            u1_r = u1[r_i]
            u0_s = u0[s_i]
            u1_s = u1[s_i]
            u0_r = torch.where(r_conj.unsqueeze(1), u0_r.conj(), u0_r)
            u1_r = torch.where(r_conj.unsqueeze(1), u1_r.conj(), u1_r)
            u0_s = torch.where(s_conj.unsqueeze(1), u0_s.conj(), u0_s)
            u1_s = torch.where(s_conj.unsqueeze(1), u1_s.conj(), u1_s)

            p0 = (s_vec * u0_r).sum(dim=1)
            p1 = (s_vec * u1_r).sum(dim=1)
            q0 = (u0_ell.conj() * u0_s).sum(dim=1)
            q1 = (u1_ell.conj() * u0_s).sum(dim=1) + (u0_ell.conj() * u1_s).sum(dim=1)
            q2 = (u1_ell.conj() * u1_s).sum(dim=1)

            coeff1 = p1 * q0 + p0 * q1
            coeff_r_ell = p1 * (u1_ell.conj() * u0_s).sum(dim=1)
            coeff_r_s = p1 * (u0_ell.conj() * u1_s).sum(dim=1)
            coeff_ell_s = p0 * q2
            coeff2 = coeff_r_ell + coeff_r_s + coeff_ell_s
            coeff3 = p1 * q2
            b_coeffs += 2.0 * torch.stack(
                [
                    (-ell2 * coeff1.imag).sum(),
                    (-ell2 * coeff2.imag).sum(),
                    (-ell2 * coeff3.imag).sum(),
                ]
            )

            def add_matrix_terms(left_group: torch.Tensor, right_group: torch.Tensor, term: torch.Tensor) -> None:
                mask = (left_group >= 0) & (right_group >= 0)
                if bool(mask.any()):
                    values = 2.0 * (-ell2[mask] * term[mask].imag)
                    quadratic_matrix.index_put_((left_group[mask], right_group[mask]), values, accumulate=True)

            add_matrix_terms(r_group, ell_group, coeff_r_ell)
            add_matrix_terms(r_group, s_group, coeff_r_s)
            add_matrix_terms(ell_group, s_group, coeff_ell_s)

        ell_idxs.clear()
        ell2s.clear()
        r_idxs.clear()
        r_conjs.clear()
        s_idxs.clear()
        s_conjs.clear()
        s_vecs.clear()

    def append_streamed(ell: tuple[int, int, int], r: tuple[int, int, int], s: tuple[int, int, int]) -> None:
        nonlocal total_count, next_progress
        append_triad(ell, r, s, pos_index, ell_idxs, ell2s, r_idxs, r_conjs, s_idxs, s_conjs, s_vecs)
        total_count += 1
        if len(ell_idxs) >= batch_triads:
            flush_batch()
        if progress_triads > 0 and total_count >= next_progress:
            if worker_count == 1:
                print(f"streamed_release_triads={total_count}", flush=True)
            else:
                print(
                    f"release_worker={worker_index + 1}/{worker_count} streamed_release_triads={total_count}",
                    flush=True,
                )
            next_progress += progress_triads

    def use_outer_index(index: int) -> bool:
        return index % worker_count == worker_index

    for ell_index, ell in enumerate(active_pos):
        if not use_outer_index(ell_index):
            continue
        for s in selected_full:
            r = subtract_modes(ell, s)
            if r in active_full_set:
                append_streamed(ell, r, s)

    for ell_index, ell in enumerate(active_pos):
        if not use_outer_index(ell_index):
            continue
        for r in selected_full:
            s = subtract_modes(ell, r)
            if s in all_full_set:
                append_streamed(ell, r, s)

    for ell_index, ell in enumerate(selected_pos):
        if not use_outer_index(ell_index):
            continue
        for r in all_full:
            s = subtract_modes(ell, r)
            if s in all_full_set:
                append_streamed(ell, r, s)

    flush_batch()
    return (b_coeffs / denominator).cpu().numpy(), (quadratic_matrix / denominator).cpu().numpy(), total_count


def write_quadratic_matrix_csv(path: Path, shells: list[int], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_shell", "col_shell", "quadratic_ratio_coefficient"])
        for row_index, row_shell in enumerate(shells):
            for col_index, col_shell in enumerate(shells):
                writer.writerow([row_shell, col_shell, f"{matrix[row_index, col_index]:.17g}"])


def print_quadratic_matrix_summary(shells: list[int], matrix: np.ndarray, top_pairs: int) -> None:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    row_abs = np.sum(np.abs(matrix), axis=1)
    print(f"quadratic_matrix_sum={float(np.sum(matrix)):.17g}")
    print(f"quadratic_sym_min_eigenvalue={float(eigenvalues[0]):.17g}")
    print(f"quadratic_sym_max_eigenvalue={float(eigenvalues[-1]):.17g}")
    print(f"quadratic_abs_row_sum_max={float(np.max(row_abs)):.17g}")
    ranked: list[tuple[float, float, int, int]] = []
    for row_index, row_shell in enumerate(shells):
        for col_index, col_shell in enumerate(shells):
            value = float(matrix[row_index, col_index])
            ranked.append((abs(value), value, row_shell, col_shell))
    ranked.sort(reverse=True)
    print("top_quadratic_pairs abs_value value row_shell col_shell")
    for abs_value, value, row_shell, col_shell in ranked[:top_pairs]:
        print(f"{abs_value:.9e} {value:.9e} {row_shell} {col_shell}")


def load_direction_cache(
    path: Path,
    selected_shells: list[int],
    selected_pos_count: int,
) -> tuple[np.ndarray, float, float, float, float, float, int] | None:
    if not path.exists():
        return None
    with np.load(path) as data:
        cached_shells = [int(value) for value in data["selected_shells"]]
        direction = data["direction"]
        if cached_shells != selected_shells or direction.shape != (selected_pos_count, 4):
            raise ValueError(
                f"direction cache {path} does not match selected shells/count: "
                f"cached_shells={cached_shells} selected_shells={selected_shells} "
                f"direction_shape={direction.shape} selected_pos_count={selected_pos_count}"
            )
        return (
            direction,
            float(data["x2"]),
            float(data["d2"]),
            float(data["linear_contribution"]),
            float(data["delta_x_fraction"]),
            float(data["delta_d_fraction"]),
            int(data["one_high_triad_count"]),
        )


def load_direction_cache_parts(
    paths: list[Path],
    selected_shells: list[int],
    selected_pos: list[tuple[int, int, int]],
) -> tuple[np.ndarray, float, float, float, float, float, int]:
    selected_shell_set = set(selected_shells)
    target_index = {mode: index for index, mode in enumerate(selected_pos)}
    direction = np.zeros((len(selected_pos), 4), dtype=np.float64)
    covered_shells: set[int] = set()
    x2_value = d2_value = None
    linear_contribution = 0.0
    delta_x_fraction = 0.0
    delta_d_fraction = 0.0
    one_high_triad_count = 0
    for path in paths:
        with np.load(path) as data:
            part_shells = [int(value) for value in data["selected_shells"]]
            part_shell_set = set(part_shells)
            if not part_shell_set <= selected_shell_set:
                raise ValueError(f"direction cache {path} has shells outside requested set: {part_shells}")
            part_direction = data["direction"]
            part_pos, _ = modes_on_shells(part_shell_set, max(part_shells))
            if part_direction.shape != (len(part_pos), 4):
                raise ValueError(
                    f"direction cache {path} direction shape {part_direction.shape} "
                    f"does not match reconstructed mode count {len(part_pos)}"
                )
            for part_index, mode in enumerate(part_pos):
                direction[target_index[mode]] = part_direction[part_index]
            part_x2 = float(data["x2"])
            part_d2 = float(data["d2"])
            if x2_value is None:
                x2_value = part_x2
                d2_value = part_d2
            elif abs(x2_value - part_x2) > 1e-6 or abs(d2_value - part_d2) > 1e-4:
                raise ValueError(f"direction cache {path} has incompatible invariants")
            linear_contribution += float(data["linear_contribution"])
            delta_x_fraction += float(data["delta_x_fraction"])
            delta_d_fraction += float(data["delta_d_fraction"])
            one_high_triad_count += int(data["one_high_triad_count"])
            covered_shells |= part_shell_set
    missing_shells = selected_shell_set - covered_shells
    if missing_shells:
        raise ValueError(f"direction cache parts do not cover shells: {sorted(missing_shells)}")
    if x2_value is None or d2_value is None:
        raise ValueError("no direction cache parts supplied")
    return direction, x2_value, d2_value, linear_contribution, delta_x_fraction, delta_d_fraction, one_high_triad_count


def save_direction_cache(
    path: Path,
    selected_shells: list[int],
    direction: np.ndarray,
    x2: float,
    d2: float,
    linear_contribution: float,
    delta_x_fraction: float,
    delta_d_fraction: float,
    one_high_triad_count: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        selected_shells=np.asarray(selected_shells, dtype=np.int64),
        direction=direction,
        x2=np.asarray(x2, dtype=np.float64),
        d2=np.asarray(d2, dtype=np.float64),
        linear_contribution=np.asarray(linear_contribution, dtype=np.float64),
        delta_x_fraction=np.asarray(delta_x_fraction, dtype=np.float64),
        delta_d_fraction=np.asarray(delta_d_fraction, dtype=np.float64),
        one_high_triad_count=np.asarray(one_high_triad_count, dtype=np.int64),
    )


def ratio_from_coeffs(
    scale: float,
    base_ratio: float,
    numerator_coeffs: np.ndarray,
    delta_x_fraction: float,
    delta_d_fraction: float,
) -> float:
    numerator = (
        base_ratio
        + numerator_coeffs[0] * scale
        + numerator_coeffs[1] * scale * scale
        + numerator_coeffs[2] * scale * scale * scale
    )
    denominator = (1.0 + scale * scale * delta_x_fraction) * np.sqrt(
        1.0 + scale * scale * delta_d_fraction
    )
    return float(numerator / denominator)


def optimize_scale(
    base_ratio: float,
    numerator_coeffs: np.ndarray,
    delta_x_fraction: float,
    delta_d_fraction: float,
    scale_max: float,
) -> tuple[float, float]:
    def objective(scale: float) -> float:
        return -ratio_from_coeffs(scale, base_ratio, numerator_coeffs, delta_x_fraction, delta_d_fraction)

    result = minimize_scalar(objective, bounds=(0.0, scale_max), method="bounded", options={"xatol": 1e-11})
    candidates = [(0.0, ratio_from_coeffs(0.0, base_ratio, numerator_coeffs, delta_x_fraction, delta_d_fraction))]
    candidates.append((scale_max, ratio_from_coeffs(scale_max, base_ratio, numerator_coeffs, delta_x_fraction, delta_d_fraction)))
    if result.success:
        candidates.append((float(result.x), -float(result.fun)))
    return max(candidates, key=lambda item: item[1])


def selected_shells_from_optimizer(
    csv_path: Path,
    base_s2: int,
    rank_start: int,
    top_shells: int,
) -> tuple[list[int], dict[int, float], float, float, float]:
    x2, d2, base_ratio, rows = load_shell_norms(csv_path, base_s2)
    result = optimize(rows, x2, d2, base_ratio)
    shells = result["shells"]
    t = result["t"]
    contribution = result["contribution"]
    order = np.argsort(contribution)[::-1]
    first_index = rank_start - 1
    if first_index < 0:
        raise ValueError("rank_start must be >= 1")
    selected_indices = order[first_index : first_index + top_shells]
    selected = [int(shells[index]) for index in selected_indices]
    t_by_shell = {int(shells[index]): float(t[index]) for index in selected_indices}
    return selected, t_by_shell, x2, d2, base_ratio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-s2", type=int, default=565)
    parser.add_argument("--base-coeffs", type=Path, default=Path("scripts/results/release_coeffs/release_s565_from550.npy"))
    parser.add_argument("--coeff-dir", type=Path, default=DEFAULT_COEFF_DIR)
    parser.add_argument("--csv-path", type=Path, default=Path("scripts/results/cstar_one_high_annulus_s565.csv"))
    parser.add_argument("--active-ratio", type=float, default=0.31186354793003834)
    parser.add_argument("--top-shells", type=int, default=3)
    parser.add_argument("--rank-start", type=int, default=1)
    parser.add_argument("--scales", type=float, nargs="*", default=[0.25, 0.5, 0.75, 1.0, 1.25])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--one-high-batch-triads", type=int, default=1_000_000)
    parser.add_argument("--one-high-progress-triads", type=int, default=5_000_000)
    parser.add_argument("--one-high-workers", type=int, default=1)
    parser.add_argument("--release-batch-triads", type=int, default=1_000_000)
    parser.add_argument("--release-progress-triads", type=int, default=5_000_000)
    parser.add_argument("--release-workers", type=int, default=1)
    parser.add_argument("--scale-max", type=float, default=3.0)
    parser.add_argument("--quadratic-matrix-csv", type=Path)
    parser.add_argument("--top-quadratic-pairs", type=int, default=12)
    parser.add_argument("--direction-cache", type=Path)
    parser.add_argument("--direction-cache-split-dir", type=Path)
    parser.add_argument("--direction-cache-parts", type=Path, nargs="*")
    parser.add_argument("--direction-only", action="store_true")
    args = parser.parse_args()

    selected_shells, t_by_shell, x2_csv, d2_csv, base_ratio_csv = selected_shells_from_optimizer(
        args.csv_path, args.base_s2, args.rank_start, args.top_shells
    )
    print(f"selected_shells={selected_shells}", flush=True)
    selected_set = set(selected_shells)
    cap_s2 = max(selected_shells)
    print(f"loading modes up to S^2={cap_s2}", flush=True)
    active_pos = positive_modes(args.base_s2)
    selected_pos, selected_full = modes_on_shells(selected_set, cap_s2)
    print(
        f"active_positive_modes={len(active_pos)} selected_positive_modes={len(selected_pos)} "
        f"selected_full_modes={len(selected_full)}",
        flush=True,
    )
    all_pos = active_pos + selected_pos
    pos_index = {mode: index for index, mode in enumerate(all_pos)}
    e1, e2, k2 = basis_arrays(all_pos)
    active_coeffs = load_base_coeffs(args.base_s2, args.coeff_dir, args.base_coeffs)
    device = torch.device(args.device)

    cached_direction = None
    if args.direction_cache_parts:
        print(f"loading direction_cache_parts={args.direction_cache_parts}", flush=True)
        cached_direction = load_direction_cache_parts(args.direction_cache_parts, selected_shells, selected_pos)
    elif args.direction_cache is not None:
        cached_direction = load_direction_cache(args.direction_cache, selected_shells, len(selected_pos))
        if cached_direction is not None:
            print(f"loading direction_cache={args.direction_cache}", flush=True)

    if cached_direction is None:
        print("streaming one-high selected triads", flush=True)
        x2, d2, gradient, one_high_triad_count = compute_selected_inactive_gradient_streaming(
            args.base_s2,
            active_pos,
            selected_pos,
            selected_full,
            pos_index,
            active_coeffs,
            e1,
            e2,
            k2,
            device,
            args.one_high_batch_triads,
            args.one_high_progress_triads,
            args.one_high_workers,
        )
        print(f"one_high_triads={one_high_triad_count}; gradient complete", flush=True)
        if abs(x2 - x2_csv) > 1e-6 or abs(d2 - d2_csv) > 1e-4 or abs(args.active_ratio - base_ratio_csv) > 1e-12:
            print("warning: invariant mismatch against CSV optimizer inputs")

        direction = np.zeros_like(gradient)
        linear_contribution = 0.0
        delta_x_fraction = 0.0
        delta_d_fraction = 0.0
        split_cache_rows: list[tuple[int, np.ndarray, float, float, float]] = []
        for shell_value in selected_shells:
            indices = [index for index, mode in enumerate(selected_pos) if shell(mode) == shell_value]
            shell_gradient = gradient[indices]
            shell_norm = float(np.linalg.norm(shell_gradient))
            t_value = t_by_shell[shell_value]
            if shell_norm == 0.0:
                split_cache_rows.append((shell_value, np.zeros((len(indices), 4), dtype=np.float64), 0.0, 0.0, 0.0))
                continue
            shell_direction = t_value * shell_gradient / shell_norm
            direction[indices] = shell_direction
            shell_linear = t_value * shell_norm
            shell_delta_x = 2.0 * shell_value * t_value * t_value / x2
            shell_delta_d = 2.0 * shell_value * shell_value * t_value * t_value / d2
            linear_contribution += shell_linear
            delta_x_fraction += shell_delta_x
            delta_d_fraction += shell_delta_d
            split_cache_rows.append((shell_value, shell_direction, shell_linear, shell_delta_x, shell_delta_d))
        if args.direction_cache is not None:
            save_direction_cache(
                args.direction_cache,
                selected_shells,
                direction,
                x2,
                d2,
                linear_contribution,
                delta_x_fraction,
                delta_d_fraction,
                one_high_triad_count,
            )
            print(f"saved direction_cache={args.direction_cache}", flush=True)
        if args.direction_cache_split_dir is not None:
            args.direction_cache_split_dir.mkdir(parents=True, exist_ok=True)
            for rank_offset, (shell_value, shell_direction, shell_linear, shell_delta_x, shell_delta_d) in enumerate(
                split_cache_rows
            ):
                rank_value = args.rank_start + rank_offset
                path = args.direction_cache_split_dir / f"cstar_annulus_direction_rank{rank_value:04d}_shell{shell_value}.npz"
                np.savez(
                    path,
                    selected_shells=np.asarray([shell_value], dtype=np.int64),
                    direction=shell_direction,
                    x2=np.asarray(x2, dtype=np.float64),
                    d2=np.asarray(d2, dtype=np.float64),
                    linear_contribution=np.asarray(shell_linear, dtype=np.float64),
                    delta_x_fraction=np.asarray(shell_delta_x, dtype=np.float64),
                    delta_d_fraction=np.asarray(shell_delta_d, dtype=np.float64),
                    one_high_triad_count=np.asarray(0, dtype=np.int64),
                    shared_one_high_triad_count=np.asarray(one_high_triad_count, dtype=np.int64),
                )
            print(
                f"saved direction_cache_split_dir={args.direction_cache_split_dir} "
                f"split_caches={len(split_cache_rows)}",
                flush=True,
            )
    else:
        (
            direction,
            x2,
            d2,
            linear_contribution,
            delta_x_fraction,
            delta_d_fraction,
            one_high_triad_count,
        ) = cached_direction
        if abs(x2 - x2_csv) > 1e-6 or abs(d2 - d2_csv) > 1e-4 or abs(args.active_ratio - base_ratio_csv) > 1e-12:
            print("warning: cached invariant mismatch against CSV optimizer inputs")

    print(f"active_modes={len(active_pos)} selected_positive_modes={len(selected_pos)}")
    print(f"one_high_triads={one_high_triad_count}")
    print(f"x2={x2:.17g} d2={d2:.17g} base_ratio={args.active_ratio:.17g}")
    print(f"linear_contribution={linear_contribution:.17g}")
    print(f"delta_X2_fraction_at_scale1={delta_x_fraction:.17g}")
    print(f"delta_D2_fraction_at_scale1={delta_d_fraction:.17g}")
    if args.direction_only:
        print("direction_only=true; skipping release polynomial", flush=True)
        return

    print("streaming selected release polynomial", flush=True)
    numerator_coeffs, quadratic_matrix, release_triad_count = eval_selected_release_polynomial_streaming(
        args.base_s2,
        active_pos,
        selected_shells,
        selected_pos,
        selected_full,
        pos_index,
        active_coeffs,
        direction,
        e1,
        e2,
        k2,
        device,
        args.release_batch_triads,
        args.release_progress_triads,
        args.release_workers,
    )
    if args.quadratic_matrix_csv is not None:
        write_quadratic_matrix_csv(args.quadratic_matrix_csv, selected_shells, quadratic_matrix)
        print(f"quadratic_matrix_csv={args.quadratic_matrix_csv}")
    linear_coeffs = np.array([numerator_coeffs[0], 0.0, 0.0], dtype=np.float64)
    linear_best_scale, linear_best_ratio = optimize_scale(
        args.active_ratio, linear_coeffs, delta_x_fraction, delta_d_fraction, args.scale_max
    )
    nonlinear_best_scale, nonlinear_best_ratio = optimize_scale(
        args.active_ratio, numerator_coeffs, delta_x_fraction, delta_d_fraction, args.scale_max
    )
    if delta_x_fraction > 0.0 and delta_d_fraction > 0.0:
        asymptotic_ratio = numerator_coeffs[2] / (delta_x_fraction * np.sqrt(delta_d_fraction))
    else:
        asymptotic_ratio = float("nan")

    print(f"release_triads={release_triad_count}")
    print(
        "numerator_ratio_coefficients "
        f"linear={numerator_coeffs[0]:.17g} quadratic={numerator_coeffs[1]:.17g} cubic={numerator_coeffs[2]:.17g}"
    )
    print(f"linear_contribution_check_diff={numerator_coeffs[0] - linear_contribution:.9e}")
    print(f"linear_best scale={linear_best_scale:.12g} ratio={linear_best_ratio:.17g}")
    print(f"nonlinear_best scale={nonlinear_best_scale:.12g} ratio={nonlinear_best_ratio:.17g}")
    print(f"nonlinear_gain_over_linear_best={nonlinear_best_ratio - linear_best_ratio:.9e}")
    print(f"positive_scale_asymptotic_ratio={asymptotic_ratio:.17g}")
    print_quadratic_matrix_summary(selected_shells, quadratic_matrix, args.top_quadratic_pairs)
    print("scale linear_model_ratio nonlinear_ratio nonlinear_minus_linear")
    for scale in args.scales:
        linear_model = ratio_from_coeffs(
            scale, args.active_ratio, linear_coeffs, delta_x_fraction, delta_d_fraction
        )
        nonlinear = ratio_from_coeffs(
            scale, args.active_ratio, numerator_coeffs, delta_x_fraction, delta_d_fraction
        )
        print(f"{scale:.6g} {linear_model:.17g} {nonlinear:.17g} {nonlinear - linear_model:.9e}")


if __name__ == "__main__":
    main()
