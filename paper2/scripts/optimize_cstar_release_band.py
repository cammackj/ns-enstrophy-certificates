"""Optimize a fixed-active release band above the archived C* candidate.

The archived S^2=261 field is held fixed. A finite band of inactive modes is
released and optimized against the exact mixed triads involving that band. The
active-only B value is supplied through the archived high-precision ratio, so
this is a diagnostic lower-bound search rather than a certificate.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from probe_cstar_inactive_gradient import (
    DEFAULT_COEFF_DIR,
    add_modes,
    append_triad,
    basis_arrays,
    load_active_coeffs,
    mode_ref,
    negate,
    positive_modes,
    shell,
    subtract_modes,
)
from shell_decomp import _TRIAD_CHUNK, get_modes


DEFAULT_ACTIVE_RATIO = 0.30262429111965870177780360306768706234239948525554
CACHE_VERSION = 1


def load_base_coeffs(base_s2: int, coeff_dir: Path, base_coeffs_path: Path | None) -> np.ndarray:
    if base_coeffs_path is not None:
        coeffs = np.load(base_coeffs_path)
    else:
        coeffs = load_active_coeffs(coeff_dir, base_s2)
    expected = len(positive_modes(base_s2)) * 4
    if coeffs.size != expected:
        raise ValueError(f"base coefficient length {coeffs.size} does not match S^2={base_s2} length {expected}")
    return coeffs.reshape(len(positive_modes(base_s2)), 4)


def build_release_triads(
    base_s2: int,
    start_s2: int,
    target_s2: int,
    active_pos: list[tuple[int, int, int]],
    inactive_pos: list[tuple[int, int, int]],
    pos_index: dict[tuple[int, int, int], int],
) -> dict[str, np.ndarray]:
    active_full = get_modes(base_s2)
    target_full = get_modes(target_s2)
    active_full_set = set(active_full)
    inactive_full = [mode for mode in target_full if start_s2 < shell(mode) <= target_s2]
    inactive_full_set = set(inactive_full)
    all_full_set = active_full_set | inactive_full_set

    ell_idxs: list[int] = []
    ell2s: list[float] = []
    r_idxs: list[int] = []
    r_conjs: list[bool] = []
    s_idxs: list[int] = []
    s_conjs: list[bool] = []
    s_vecs: list[tuple[float, float, float]] = []

    # Active output, active r, inactive s.
    for ell in active_pos:
        for s in inactive_full:
            r = subtract_modes(ell, s)
            if r in active_full_set:
                append_triad(ell, r, s, pos_index, ell_idxs, ell2s, r_idxs, r_conjs, s_idxs, s_conjs, s_vecs)

    # Active output, inactive r, active or inactive s.
    for ell in active_pos:
        for r in inactive_full:
            s = subtract_modes(ell, r)
            if s in all_full_set:
                append_triad(ell, r, s, pos_index, ell_idxs, ell2s, r_idxs, r_conjs, s_idxs, s_conjs, s_vecs)

    # Inactive output, any target-band r/s pair.
    all_full = active_full + inactive_full
    for ell in inactive_pos:
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


def release_cache_path(cache_dir: Path, base_s2: int, start_s2: int, target_s2: int) -> Path:
    if start_s2 == base_s2:
        return cache_dir / f"release_v{CACHE_VERSION}_s{base_s2}_to_s{target_s2}.npz"
    return cache_dir / f"release_v{CACHE_VERSION}_s{base_s2}_from_s{start_s2}_to_s{target_s2}.npz"


def load_or_build_release_triads(
    base_s2: int,
    start_s2: int,
    target_s2: int,
    active_pos: list[tuple[int, int, int]],
    inactive_pos: list[tuple[int, int, int]],
    pos_index: dict[tuple[int, int, int], int],
    cache_dir: Path | None,
) -> dict[str, np.ndarray]:
    if cache_dir is None:
        return build_release_triads(base_s2, start_s2, target_s2, active_pos, inactive_pos, pos_index)

    path = release_cache_path(cache_dir, base_s2, start_s2, target_s2)
    if path.exists():
        print(f"  loading release triads: {path}", flush=True)
        with np.load(path) as data:
            return {name: data[name] for name in data.files}

    triads = build_release_triads(base_s2, start_s2, target_s2, active_pos, inactive_pos, pos_index)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"  saving release triads: {path}", flush=True)
    np.savez(path, **triads)
    return triads


class ReleaseProblem:
    def __init__(
        self,
        active_coeffs_np: np.ndarray,
        inactive_count: int,
        e1_np: np.ndarray,
        e2_np: np.ndarray,
        k2_np: np.ndarray,
        triads: dict[str, np.ndarray],
        active_ratio: float,
        device: torch.device,
    ) -> None:
        self.device = device
        self.dtype = torch.float64
        self.inactive_count = inactive_count
        self.active_coeffs = torch.tensor(active_coeffs_np, dtype=self.dtype, device=device)
        self.e1 = torch.tensor(e1_np, dtype=self.dtype, device=device)
        self.e2 = torch.tensor(e2_np, dtype=self.dtype, device=device)
        self.k2 = torch.tensor(k2_np, dtype=self.dtype, device=device)
        self.k4 = self.k2 * self.k2
        zeros3 = torch.zeros_like(self.e1)
        self.e1_c = torch.complex(self.e1, zeros3)
        self.e2_c = torch.complex(self.e2, zeros3)

        use_cuda = device.type == "cuda"

        def tensor_cpu(name: str, dtype_value: torch.dtype) -> torch.Tensor:
            tensor = torch.tensor(triads[name], dtype=dtype_value)
            return tensor.pin_memory() if use_cuda else tensor

        self.ell_idxs = tensor_cpu("ell_idxs", torch.long)
        self.ell2s = tensor_cpu("ell2s", self.dtype)
        self.r_idxs = tensor_cpu("r_idxs", torch.long)
        self.r_conjs = tensor_cpu("r_conjs", torch.bool)
        self.s_idxs = tensor_cpu("s_idxs", torch.long)
        self.s_conjs = tensor_cpu("s_conjs", torch.bool)
        s_vecs_real = torch.tensor(triads["s_vecs"], dtype=self.dtype)
        self.s_vecs = torch.complex(s_vecs_real, torch.zeros_like(s_vecs_real))
        if use_cuda:
            self.s_vecs = self.s_vecs.pin_memory()

        with torch.no_grad():
            zero = torch.zeros((inactive_count, 4), dtype=self.dtype, device=device)
            _, x2_active, d2_active = self.evaluate_components(zero)
            self.active_b = torch.tensor(
                active_ratio * float(x2_active.cpu()) * float(d2_active.cpu()) ** 0.5,
                dtype=self.dtype,
                device=device,
            )

    def velocity(self, inactive_coeffs: torch.Tensor) -> torch.Tensor:
        coeffs = torch.cat([self.active_coeffs, inactive_coeffs], dim=0)
        c1 = torch.view_as_complex(coeffs[:, :2].contiguous())
        c2 = torch.view_as_complex(coeffs[:, 2:].contiguous())
        return c1.unsqueeze(1) * self.e1_c + c2.unsqueeze(1) * self.e2_c

    def mixed_b(self, u: torch.Tensor) -> torch.Tensor:
        triad_count = int(self.ell_idxs.shape[0])
        total = torch.zeros((), dtype=self.dtype, device=self.device)
        for start in range(0, triad_count, _TRIAD_CHUNK):
            end = min(start + _TRIAD_CHUNK, triad_count)
            ell_i = self.ell_idxs[start:end].to(self.device, non_blocking=True)
            r_i = self.r_idxs[start:end].to(self.device, non_blocking=True)
            s_i = self.s_idxs[start:end].to(self.device, non_blocking=True)
            r_conj = self.r_conjs[start:end].to(self.device, non_blocking=True)
            s_conj = self.s_conjs[start:end].to(self.device, non_blocking=True)
            ell2 = self.ell2s[start:end].to(self.device, non_blocking=True)
            s_vec = self.s_vecs[start:end].to(self.device, non_blocking=True)
            u_ell = u[ell_i]
            u_r = u[r_i]
            u_s = u[s_i]
            u_r = torch.where(r_conj.unsqueeze(1), u_r.conj(), u_r)
            u_s = torch.where(s_conj.unsqueeze(1), u_s.conj(), u_s)
            s_dot_ur = (s_vec * u_r).sum(dim=1)
            uell_dot_us = (u_ell.conj() * u_s).sum(dim=1)
            total = total + 2.0 * (-ell2 * (s_dot_ur * uell_dot_us).imag).sum()
        return total

    def evaluate_components(self, inactive_coeffs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u = self.velocity(inactive_coeffs)
        amp2 = 2.0 * u.abs().pow(2).sum(dim=1)
        x2 = (self.k2 * amp2).sum()
        d2 = (self.k4 * amp2).sum()
        mixed_b = self.mixed_b(u)
        return mixed_b, x2, d2

    def ratio(self, inactive_coeffs: torch.Tensor) -> torch.Tensor:
        mixed_b, x2, d2 = self.evaluate_components(inactive_coeffs)
        return (self.active_b + mixed_b) / (x2 * d2.sqrt())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-s2", type=int, default=261)
    parser.add_argument("--start-s2", type=int, default=None)
    parser.add_argument("--target-s2", type=int, default=262)
    parser.add_argument("--coeff-dir", type=Path, default=DEFAULT_COEFF_DIR)
    parser.add_argument("--base-coeffs", type=Path, default=None)
    parser.add_argument("--active-ratio", type=float, default=DEFAULT_ACTIVE_RATIO)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--line-start", type=float, default=100.0)
    parser.add_argument("--random-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--save-path", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.target_s2 <= args.base_s2:
        raise ValueError("target-s2 must be larger than base-s2")
    start_s2 = args.base_s2 if args.start_s2 is None else args.start_s2
    if start_s2 < args.base_s2:
        raise ValueError("start-s2 must be at least base-s2")
    if args.target_s2 <= start_s2:
        raise ValueError("target-s2 must be larger than start-s2")

    device = torch.device(args.device)
    print(f"Building release band S^2={args.base_s2}, {start_s2} < |k|^2 <= {args.target_s2} on {device} ...", flush=True)
    active_pos = positive_modes(args.base_s2)
    target_pos = positive_modes(args.target_s2)
    inactive_pos = [mode for mode in target_pos if start_s2 < shell(mode) <= args.target_s2]
    all_pos = active_pos + inactive_pos
    pos_index = {mode: index for index, mode in enumerate(all_pos)}
    e1, e2, k2 = basis_arrays(all_pos)
    triads = load_or_build_release_triads(
        args.base_s2, start_s2, args.target_s2, active_pos, inactive_pos, pos_index, args.cache_dir
    )
    print(
        f"  active modes={len(active_pos)} inactive modes={len(inactive_pos)} "
        f"release triads={len(triads['ell_idxs'])}",
        flush=True,
    )

    active_coeffs = load_base_coeffs(args.base_s2, args.coeff_dir, args.base_coeffs)
    problem = ReleaseProblem(active_coeffs, len(inactive_pos), e1, e2, k2, triads, args.active_ratio, device)

    inactive = torch.zeros((len(inactive_pos), 4), dtype=torch.float64, device=device, requires_grad=True)
    initial_ratio = float(problem.ratio(inactive).detach().cpu())
    loss = -problem.ratio(inactive)
    loss.backward()
    gradient = -inactive.grad.detach().clone()
    grad_norm = gradient.norm()
    print(f"  initial ratio={initial_ratio:.17g} inactive_grad_l2={float(grad_norm.cpu()):.6e}", flush=True)

    with torch.no_grad():
        if args.random_scale > 0.0:
            rng = np.random.default_rng(args.seed)
            random_start = rng.standard_normal(size=(len(inactive_pos), 4)) * args.random_scale
            inactive.copy_(torch.tensor(random_start, dtype=torch.float64, device=device))
        elif grad_norm > 0:
            inactive.copy_(args.line_start * gradient / grad_norm)
    inactive.grad = None

    optimizer = torch.optim.LBFGS(
        [inactive],
        max_iter=args.max_iter,
        tolerance_grad=1e-14,
        tolerance_change=1e-16,
        line_search_fn="strong_wolfe",
    )

    iteration = 0

    def closure() -> torch.Tensor:
        nonlocal iteration
        optimizer.zero_grad(set_to_none=True)
        value = -problem.ratio(inactive)
        value.backward()
        iteration += 1
        if iteration == 1 or iteration % 10 == 0:
            print(f"    eval {iteration:03d}: ratio={float((-value).detach().cpu()):.17g}", flush=True)
        return value

    optimizer.step(closure)
    final_ratio = float(problem.ratio(inactive).detach().cpu())
    final_coeffs = inactive.detach().cpu().numpy()
    print(f"  final ratio={final_ratio:.17g} improvement={final_ratio - initial_ratio:.6e}")
    print(f"  inactive coeff l2={np.linalg.norm(final_coeffs):.6e} inf={np.max(np.abs(final_coeffs)):.6e}")

    if args.save_path is not None:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        coeff_by_mode = {mode: active_coeffs[index].copy() for index, mode in enumerate(active_pos)}
        for index, mode in enumerate(inactive_pos):
            coeff_by_mode[mode] = final_coeffs[index].copy()
        target_coeffs = np.zeros((len(target_pos), 4), dtype=np.float64)
        for index, mode in enumerate(target_pos):
            target_coeffs[index] = coeff_by_mode[mode]
        np.save(args.save_path, target_coeffs.reshape(-1))
        print(f"  saved combined coeffs: {args.save_path}")

    per_mode = []
    for mode_index, mode in enumerate(inactive_pos):
        block = final_coeffs[mode_index]
        per_mode.append((float(np.linalg.norm(block)), mode, block.copy()))
    per_mode.sort(reverse=True, key=lambda item: item[0])
    print("Top released inactive coefficient blocks:")
    for norm, mode, block in per_mode[:12]:
        print(f"  mode={mode} |k|2={shell(mode)} coeff_l2={norm:.6e} block={block}")


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()