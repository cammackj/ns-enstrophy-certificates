"""First-order inactive-mode probe for the full-field C* candidate.

Embeds the archived S^2=261 optimizer into a larger Galerkin cap and computes
the inactive gradient of R=B/(X^2 D). At zero inactive amplitude, only triads
with exactly one inactive factor contribute to the first derivative, so this
script enumerates just those triads instead of rebuilding the full Galerkin
problem for the target cap.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from shell_decomp import _TRIAD_CHUNK, build_div_free_basis, get_modes, pos_half


ROOT = Path(__file__).resolve().parent
DEFAULT_COEFF_DIR = ROOT / "results" / "opt_coeffs"


def positive_modes(s2_max: int) -> list[tuple[int, int, int]]:
    return pos_half(get_modes(s2_max))


def shell(mode: tuple[int, int, int]) -> int:
    return sum(component * component for component in mode)


def negate(mode: tuple[int, int, int]) -> tuple[int, int, int]:
    return (-mode[0], -mode[1], -mode[2])


def add_modes(left: tuple[int, int, int], right: tuple[int, int, int]) -> tuple[int, int, int]:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def subtract_modes(left: tuple[int, int, int], right: tuple[int, int, int]) -> tuple[int, int, int]:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def load_active_coeffs(coeff_dir: Path, base_s2: int) -> np.ndarray:
    base_pos = positive_modes(base_s2)
    return np.load(coeff_dir / f"opt_coeffs_s{base_s2}.npy").reshape(len(base_pos), 4)


def basis_arrays(pos_modes: list[tuple[int, int, int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    e1 = np.zeros((len(pos_modes), 3), dtype=np.float64)
    e2 = np.zeros((len(pos_modes), 3), dtype=np.float64)
    k2 = np.zeros(len(pos_modes), dtype=np.float64)
    for index, mode in enumerate(pos_modes):
        mode_e1, mode_e2 = build_div_free_basis(mode)
        e1[index] = mode_e1
        e2[index] = mode_e2
        k2[index] = shell(mode)
    return e1, e2, k2


def mode_ref(
    mode: tuple[int, int, int],
    pos_index: dict[tuple[int, int, int], int],
) -> tuple[int, bool]:
    if mode in pos_index:
        return pos_index[mode], False
    negative_mode = negate(mode)
    if negative_mode in pos_index:
        return pos_index[negative_mode], True
    raise KeyError(mode)


def append_triad(
    ell: tuple[int, int, int],
    r: tuple[int, int, int],
    s: tuple[int, int, int],
    pos_index: dict[tuple[int, int, int], int],
    ell_idxs: list[int],
    ell2s: list[float],
    r_idxs: list[int],
    r_conjs: list[bool],
    s_idxs: list[int],
    s_conjs: list[bool],
    s_vecs: list[tuple[float, float, float]],
) -> None:
    ell_i, ell_conj = mode_ref(ell, pos_index)
    if ell_conj:
        raise ValueError("output ell must be in the positive half")
    r_i, r_conj = mode_ref(r, pos_index)
    s_i, s_conj = mode_ref(s, pos_index)
    ell_idxs.append(ell_i)
    ell2s.append(float(shell(ell)))
    r_idxs.append(r_i)
    r_conjs.append(r_conj)
    s_idxs.append(s_i)
    s_conjs.append(s_conj)
    s_vecs.append((float(s[0]), float(s[1]), float(s[2])))


def build_one_inactive_triads(
    base_s2: int,
    target_s2: int,
    active_pos: list[tuple[int, int, int]],
    inactive_pos: list[tuple[int, int, int]],
    pos_index: dict[tuple[int, int, int], int],
) -> dict[str, np.ndarray]:
    active_full = get_modes(base_s2)
    target_full = get_modes(target_s2)
    active_full_set = set(active_full)
    active_pos_set = set(active_pos)
    inactive_full = [mode for mode in target_full if shell(mode) > base_s2]

    ell_idxs: list[int] = []
    ell2s: list[float] = []
    r_idxs: list[int] = []
    r_conjs: list[bool] = []
    s_idxs: list[int] = []
    s_conjs: list[bool] = []
    s_vecs: list[tuple[float, float, float]] = []

    # Case 1: inactive output ell, active r, active s.
    for ell in inactive_pos:
        for r in active_full:
            s = subtract_modes(ell, r)
            if s in active_full_set:
                append_triad(ell, r, s, pos_index, ell_idxs, ell2s, r_idxs, r_conjs, s_idxs, s_conjs, s_vecs)

    # Case 2: active output ell, inactive r, active s.
    for ell in active_pos:
        for r in inactive_full:
            s = subtract_modes(ell, r)
            if s in active_full_set:
                append_triad(ell, r, s, pos_index, ell_idxs, ell2s, r_idxs, r_conjs, s_idxs, s_conjs, s_vecs)

    # Case 3: active output ell, active r, inactive s. Iterate by inactive s
    # so the loop cost is O(#inactive shell * #active modes), not O(active^2).
    for s in inactive_full:
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


def compute_inactive_gradient(
    active_coeffs_np: np.ndarray,
    inactive_count: int,
    e1_np: np.ndarray,
    e2_np: np.ndarray,
    k2_np: np.ndarray,
    triads: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[float, float, np.ndarray]:
    dtype = torch.float64
    use_cuda = device.type == "cuda"
    active_coeffs = torch.tensor(active_coeffs_np, dtype=dtype, device=device)
    inactive_coeffs = torch.zeros((inactive_count, 4), dtype=dtype, device=device, requires_grad=True)
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

    def tensor_cpu(name: str, dtype_value: torch.dtype) -> torch.Tensor:
        tensor = torch.tensor(triads[name], dtype=dtype_value)
        return tensor.pin_memory() if use_cuda else tensor

    ell_idxs = tensor_cpu("ell_idxs", torch.long)
    ell2s = tensor_cpu("ell2s", dtype)
    r_idxs = tensor_cpu("r_idxs", torch.long)
    r_conjs = tensor_cpu("r_conjs", torch.bool)
    s_idxs = tensor_cpu("s_idxs", torch.long)
    s_conjs = tensor_cpu("s_conjs", torch.bool)
    s_vecs_real = torch.tensor(triads["s_vecs"], dtype=dtype)
    s_vecs = torch.complex(s_vecs_real, torch.zeros_like(s_vecs_real))
    if use_cuda:
        s_vecs = s_vecs.pin_memory()

    u_leaf = u.detach().clone().requires_grad_(True)
    b_linear = torch.zeros((), dtype=dtype, device=device)
    triad_count = int(ell_idxs.shape[0])
    for start in range(0, triad_count, _TRIAD_CHUNK):
        end = min(start + _TRIAD_CHUNK, triad_count)
        ell_i = ell_idxs[start:end].to(device, non_blocking=True)
        r_i = r_idxs[start:end].to(device, non_blocking=True)
        s_i = s_idxs[start:end].to(device, non_blocking=True)
        r_conj = r_conjs[start:end].to(device, non_blocking=True)
        s_conj = s_conjs[start:end].to(device, non_blocking=True)
        ell2 = ell2s[start:end].to(device, non_blocking=True)
        s_vec = s_vecs[start:end].to(device, non_blocking=True)

        u_ell = u_leaf[ell_i]
        u_r = u_leaf[r_i]
        u_s = u_leaf[s_i]
        u_r = torch.where(r_conj.unsqueeze(1), u_r.conj(), u_r)
        u_s = torch.where(s_conj.unsqueeze(1), u_s.conj(), u_s)
        s_dot_ur = (s_vec * u_r).sum(dim=1)
        uell_dot_us = (u_ell.conj() * u_s).sum(dim=1)
        b_chunk = 2.0 * (-ell2 * (s_dot_ur * uell_dot_us).imag).sum()
        b_linear = b_linear + b_chunk.detach()
        b_chunk.backward()

        del ell_i, r_i, s_i, r_conj, s_conj, ell2, s_vec
        del u_ell, u_r, u_s, s_dot_ur, uell_dot_us, b_chunk

    u.backward(gradient=u_leaf.grad)
    gradient = inactive_coeffs.grad.detach() / denominator
    return float(x2.cpu()), float(d2.cpu()), gradient.cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-s2", type=int, default=261)
    parser.add_argument("--target-s2", type=int, default=262)
    parser.add_argument("--coeff-dir", type=Path, default=DEFAULT_COEFF_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.target_s2 <= args.base_s2:
        raise ValueError("target-s2 must be larger than base-s2")

    device = torch.device(args.device)
    print(
        f"Building one-inactive triads for S^2={args.base_s2} -> {args.target_s2} on {device} ...",
        flush=True,
    )
    active_pos = positive_modes(args.base_s2)
    target_pos = positive_modes(args.target_s2)
    inactive_pos = [mode for mode in target_pos if shell(mode) > args.base_s2]
    all_pos = active_pos + inactive_pos
    pos_index = {mode: index for index, mode in enumerate(all_pos)}
    e1, e2, k2 = basis_arrays(all_pos)
    triads = build_one_inactive_triads(args.base_s2, args.target_s2, active_pos, inactive_pos, pos_index)
    print(
        f"  active modes={len(active_pos)} inactive modes={len(inactive_pos)} "
        f"one-inactive triads={len(triads['ell_idxs'])}",
        flush=True,
    )

    active_coeffs = load_active_coeffs(args.coeff_dir, args.base_s2)
    x2, d2, inactive_gradient = compute_inactive_gradient(
        active_coeffs, len(inactive_pos), e1, e2, k2, triads, device
    )
    print(f"Embedded X2={x2:.17g} D2={d2:.17g}")
    print(
        f"inactive grad inf={np.max(np.abs(inactive_gradient)):.6e} "
        f"l2={np.linalg.norm(inactive_gradient):.6e}"
    )

    per_mode = []
    for mode_index, mode in enumerate(inactive_pos):
        block = inactive_gradient[mode_index]
        per_mode.append((float(np.linalg.norm(block)), mode, block.copy()))
    per_mode.sort(reverse=True, key=lambda item: item[0])
    print("Top inactive mode gradient blocks:")
    for norm, mode, block in per_mode[:12]:
        print(f"  mode={mode} |k|2={shell(mode)} grad_l2={norm:.6e} block={block}")


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()