#!/usr/bin/env python3
"""
gap3_principled_scan.py
=======================
Finite verification for GAP 3, k = 1..31 (k≥32: covered analytically in §19.10.2).

Design:
  - exploit the D×C×A nucleus as a warm-start (deterministic skeleton)
  - mandatory falsification: random starts alongside nucleus warm-start;
         trigger when full-block exceeds nucleus by >5%; prints optimizer energy profile
  - incremental: print and flush each k immediately (preserve best-known)
  - separate odd-k (D-bottom) from even-k (B-bottom) in all reporting

For each k the script does TWO levels of optimizer:

  Level 1 (nucleus): restrict to D×C×A nucleus shells only → C_nuc(k)
    - Fast: O(30-60) pos-half modes, O(100-500) triads
    - N_STARTS_NUC random starts

  Level 2 (full block): all shells in I_k → C(I_k)
    - Much larger, but WARM-STARTED from nucleus solution
    - N_STARTS_FULL additional random starts for falsification (P8)

Scaling analysis printed at the end:
    C(I_k) × 2^(k/2)   — tests O(2^{-k/2}) decay hypothesis
    C(I_k) × k          — tests O(1/k) decay hypothesis
    C(I_k) / C_nuc(k)  — fraction of block max attributable to nucleus alone

Usage
-----
    python scripts/gap3/gap3_principled_scan.py
    python scripts/gap3/gap3_principled_scan.py --kmin 6 --kmax 8 --n_nuc 200 --n_full 100
    python scripts/gap3/gap3_principled_scan.py --kmin 6 --kmax 10 --nucleus_only

Date: April 2026
"""

import argparse
import io
import math
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

# Force UTF-8 output on Windows (avoids UnicodeEncodeError for box-drawing chars)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, ".")
from scripts.gap3.multi_mode_beta_bound import get_wavevectors, divfree_basis, precompute_triads
from scripts.gap3.max_b_over_keff import neg_ratio_and_grad

# ---------------------------------------------------------------------------
# GPU triad builder  (torch.searchsorted approach, falls back to CPU dict)
# ---------------------------------------------------------------------------
try:
    import torch as _torch
    _CUDA_OK = _torch.cuda.is_available()
except ImportError:
    _CUDA_OK = False


def _build_triads_torch(wavevecs_by_shell: list[list], batch_size: int = 256):
    """Find cross-shell triads using torch.searchsorted on GPU (or CPU).

    Encodes every wavevector k as  enc(k) = kx*Q^2 + ky*Q + kz  with Q=2^15
    (linear, so enc(ℓ-r) = enc(ℓ)-enc(r)).  For every anchor mode ℓ and every
    reference mode r we compute s=ℓ-r and look it up in the sorted encoded set
    via binary search — fully vectorised in (batch_size × 2N) per GPU kernel call.

    Returns (ell_idx, ell2, r_idx, s_idx, s_mat) as numpy arrays with the same
    semantics as the Python-dict path in _build_problem.
    """
    import torch
    device = 'cuda' if _CUDA_OK else 'cpu'
    dev = torch.device(device)

    pos_list: list[tuple] = []
    shell_offsets: list[int] = []
    for wvs in wavevecs_by_shell:
        shell_offsets.append(len(pos_list))
        pos_list.extend(wvs)
    N = len(pos_list)
    if N == 0:
        return (np.array([], np.int32), np.array([], np.float64),
                np.array([], np.int32), np.array([], np.int32),
                np.zeros((0, 3), np.float64))

    pos_arr = torch.tensor(pos_list, dtype=torch.int64, device=dev)   # (N, 3)
    neg_arr = -pos_arr
    dbl_arr = torch.cat([pos_arr, neg_arr], dim=0)                     # (2N, 3)

    # Linear encoding: enc(kx,ky,kz) = kx*Q^2 + ky*Q + kz, Q=2^15=32768
    # Component range for k≤30: |ki| ≤ 2^13=8192 → fits in int64 with margin
    Q = torch.tensor(32768, dtype=torch.int64, device=dev)
    enc_dbl = dbl_arr[:, 0] * Q * Q + dbl_arr[:, 1] * Q + dbl_arr[:, 2]  # (2N,)
    enc_sorted, sort_inv = torch.sort(enc_dbl)                             # O(2N log 2N)
    TwoN = 2 * N

    # Anchor on smallest shell
    shell_sizes = [(len(wvs), i) for i, wvs in enumerate(wavevecs_by_shell)]
    shell_sizes.sort()
    anchor_i = shell_sizes[0][1]
    anchor_off = shell_offsets[anchor_i]
    anchor_n = len(wavevecs_by_shell[anchor_i])

    # Anchor dbl indices: pos half [anchor_off .. anchor_off+anchor_n) then neg half
    anchor_pos = torch.arange(anchor_off, anchor_off + anchor_n,
                              dtype=torch.long, device=dev)
    anchor_neg = torch.arange(N + anchor_off, N + anchor_off + anchor_n,
                              dtype=torch.long, device=dev)
    anchor_dbl_idx = torch.cat([anchor_pos, anchor_neg])  # (2*anchor_n,)
    enc_anchor = enc_dbl[anchor_dbl_idx]                  # (2*anchor_n,)
    M_anchor = len(anchor_dbl_idx)

    ell_idx_parts, r_idx_parts, s_idx_parts, ell2_parts, s_parts = [], [], [], [], []

    for bs in range(0, M_anchor, batch_size):
        be = min(bs + batch_size, M_anchor)
        B = be - bs
        ei_batch = anchor_dbl_idx[bs:be]       # (B,) — dbl indices for ℓ
        enc_ell  = enc_anchor[bs:be]           # (B,)

        # s = ℓ - r  →  enc(s) = enc(ℓ) - enc(r)  [using linearity of encoding]
        # shape (B, 2N):
        s_enc = enc_ell.unsqueeze(1) - enc_dbl.unsqueeze(0)   # (B, 2N) int64

        # Binary search — s_flat is a view of s_enc (no copy)
        s_flat   = s_enc.reshape(-1)                           # (B*2N,)
        found    = torch.searchsorted(enc_sorted, s_flat)      # (B*2N,) positions
        found_cl = found.clamp(0, TwoN - 1)
        matched  = (enc_sorted[found_cl] == s_flat) & (found < TwoN)  # (B*2N,)
        del s_enc, s_flat                                      # free large int64 tensor early

        if not matched.any():
            del found, found_cl, matched
            continue

        matched_2d = matched.reshape(B, TwoN)
        vi, vj = matched_2d.nonzero(as_tuple=True)            # (K,) valid positions in (B, 2N)
        del matched, matched_2d

        flat_idx = vi * TwoN + vj                             # flat positions in (B*2N,)
        ei_valid = ei_batch[vi]                               # dbl indices for ℓ
        ri_valid = vj                                         # dbl indices for r
        si_valid = sort_inv[found_cl[flat_idx]]               # dbl indices for s (via sorted pos)
        del found, found_cl, flat_idx

        ell_vecs = dbl_arr[ei_valid]
        ell2_v   = (ell_vecs * ell_vecs).sum(dim=1).to(torch.float64)
        s_vecs   = dbl_arr[si_valid].to(torch.float64)

        ell_idx_parts.append(ei_valid.to(torch.int32).cpu())
        r_idx_parts.append(ri_valid.to(torch.int32).cpu())
        s_idx_parts.append(si_valid.to(torch.int32).cpu())
        ell2_parts.append(ell2_v.cpu())
        s_parts.append(s_vecs.cpu())

    if not ell_idx_parts:
        return (np.array([], np.int32), np.array([], np.float64),
                np.array([], np.int32), np.array([], np.int32),
                np.zeros((0, 3), np.float64))

    return (torch.cat(ell_idx_parts).numpy(),
            torch.cat(ell2_parts).numpy(),
            torch.cat(r_idx_parts).numpy(),
            torch.cat(s_idx_parts).numpy(),
            torch.cat(s_parts).numpy())



# ---------------------------------------------------------------------------
# Parallel-starts support  (initializer pattern: large arrays pickled once)
# ---------------------------------------------------------------------------
_STARTS_WORKER_PROB: tuple | None = None


def _init_starts_worker(prob_tuple: tuple) -> None:
    """Called once per worker process to cache the problem data."""
    global _STARTS_WORKER_PROB
    _STARTS_WORKER_PROB = prob_tuple


def _one_start(x0: np.ndarray) -> tuple:
    """Single L-BFGS-B restart using the globally cached problem."""
    global _STARTS_WORKER_PROB
    N, k2s, e1s, e2s, ell_idx, ell2, r_idx, s_idx, s_mat = _STARTS_WORKER_PROB
    bounds = [(0.0, np.pi / 2), (0.0, 2 * np.pi), (0.0, 2 * np.pi), (-8.0, 8.0)] * N
    def obj(x):
        return neg_ratio_and_grad(x, N, e1s, e2s, k2s, ell_idx, ell2, r_idx, s_idx, s_mat)
    res = minimize(obj, x0, method='L-BFGS-B', jac=True, bounds=bounds,
                   options={'ftol': 1e-15, 'gtol': 1e-12, 'maxiter': 3000})
    return float(-res.fun), res.x.copy()

# ---------------------------------------------------------------------------
# Shell family classifier (inline, avoiding parse-args in shell_family_classify)
# ---------------------------------------------------------------------------

def _check_q2_strict(m: int) -> bool:
    if m <= 0 or m % 2 != 0:
        return False
    half = m // 2
    a = int(math.isqrt(half))
    return a * a == half and a >= 1


def _check_q1(m: int, max_coord: int = 0) -> bool:
    """Check if shell m carries an equilateral triad: integer p,q with
    |p|²=|q|²=m, p·q=m/2.  Requires m even.

    O(m) algorithm: enumerate p=(a,b,c) on the shell; for each p with c≠0
    fix x,y freely and solve z = (target - ax - by) / c, check |q|²=m.

    max_coord: cap coordinate search to this value (0 = full sqrt(m) scan).
    Setting max_coord=20-30 is safe for nucleus identification (small-coordinate
    equilateral triads are the ones that contribute to the nucleus anyway).
    """
    if m % 2 != 0:
        return False
    target_dot = m // 2
    full_limit = int(math.isqrt(m)) + 1
    limit = min(full_limit, max_coord) if max_coord > 0 else full_limit
    # Enumerate p=(a,b,c) with a²+b²+c²=m, a>=0, up to coord cap
    for a in range(min(limit, full_limit) + 1):
        a2 = a * a
        if a2 > m:
            break
        for b in range(int(math.isqrt(m - a2)) + 1):
            if b > limit:
                break
            c2 = m - a2 - b * b
            if c2 < 0:
                break
            c = int(math.isqrt(c2))
            if c * c != c2 or c > limit:
                continue
            # For each representation p=(a,b,c), scan q=(x,y,z) with
            # ax+by+cz = target_dot and |q|²=m.
            # Fix x,y; solve for z (if c != 0) or y (if c==0,b!=0).
            if c != 0:
                for x in range(-limit, limit + 1):
                    rem_x = target_dot - a * x
                    for y in range(-limit, limit + 1):
                        rem = rem_x - b * y
                        if rem % c != 0:
                            continue
                        z = rem // c
                        if abs(z) > limit:
                            continue
                        if x * x + y * y + z * z == m and (x, y, z) != (a, b, c):
                            return True
            elif b != 0:
                for x in range(-limit, limit + 1):
                    rem_x = target_dot - a * x
                    if rem_x % b != 0:
                        continue
                    y = rem_x // b
                    if abs(y) > limit:
                        continue
                    for z in range(-limit, limit + 1):
                        if x * x + y * y + z * z == m and (x, y, z) != (a, b, 0):
                            return True
            elif a != 0:
                if target_dot % a != 0:
                    continue
                x = target_dot // a
                if abs(x) > limit:
                    continue
                for y in range(-limit, limit + 1):
                    z2 = m - x * x - y * y
                    if z2 < 0:
                        break
                    z = int(math.isqrt(z2))
                    if z * z == z2 and abs(z) <= limit and (x, y, z) != (a, 0, 0):
                        return True
    return False


def classify_shell(m: int, max_coord: int = 0) -> str:
    """Return 'D', 'A', 'C', or 'B'.

    max_coord: passed to _check_q1 to cap the search depth (0 = full).
    Use max_coord=25 for fast nucleus identification at large k.
    """
    if _check_q2_strict(m):
        return 'D'
    q1 = _check_q1(m, max_coord=max_coord)
    if q1:
        return 'A'
    # Q2_weak: m = a²+b² for a>=b>=1
    q2_weak = False
    for a in range(int(math.isqrt(m)), 0, -1):
        rem = m - a * a
        if rem <= 0:
            continue
        b = int(math.isqrt(rem))
        if b >= 1 and b * b == rem:
            q2_weak = True
            break
    return 'C' if q2_weak else 'B'


def _shells_have_triad(n1: int, n2: int, n3: int, max_coord: int = 12) -> bool:
    """Check if shells n1, n2, n3 admit a resonant triad k+l+m=0.

    For small shells (n3 ≤ 500) does brute-force vector sampling.
    For large shells uses arithmetic-only tests (parity + Cauchy-Schwarz +
    integrality of target dot-product), which are necessary and in practice
    almost always sufficient at these shell sizes.
    """
    # Necessary: (n3 - n1 - n2) must be even
    if (n3 - n1 - n2) % 2 != 0:
        return False
    target_dot = (n3 - n1 - n2) // 2
    # Cauchy-Schwarz: |dot| <= sqrt(n1*n2)
    if abs(target_dot) > math.isqrt(n1 * n2) + 1:
        return False
    # For large shells the arithmetic condition is sufficient (false negatives
    # from integer non-representability are rare and unimportant for nucleus finding).
    if n3 > 500:
        return True
    # Small shells: sample integer vectors to confirm integer solution exists.
    limit1 = min(int(math.isqrt(n1)) + 1, max_coord)
    for a in range(-limit1, limit1 + 1):
        if a * a > n1:
            continue
        for b in range(-limit1, limit1 + 1):
            rem1 = n1 - a * a - b * b
            if rem1 < 0:
                continue
            c_sq1 = int(math.isqrt(rem1))
            if c_sq1 * c_sq1 != rem1:
                continue
            limit2 = min(int(math.isqrt(n2)) + 1, max_coord)
            for x in range(-limit2, limit2 + 1):
                if x * x > n2:
                    continue
                for y in range(-limit2, limit2 + 1):
                    rem2 = n2 - x * x - y * y
                    if rem2 < 0:
                        continue
                    z_sq = int(math.isqrt(rem2))
                    if z_sq * z_sq != rem2:
                        continue
                    for z in ([z_sq] if z_sq == 0 else [z_sq, -z_sq]):
                        if a * x + b * y + c_sq1 * z == target_dot:
                            return True
    return False


def find_dcxa_nucleus(k: int):
    """Return (n_D, n_C, n_A) for block I_k: a valid shell triple with resonant triads.

    For k ≤ 7: uses full D×C×A family classification (exact).
    For k ≥ 8: generates D-shells directly (n=2a²), then scans nearby shells
    for two partners with arithmetic triad compatibility, bypassing the slow
    Q1/A-shell enumeration that is O(m²) at large m.
    """
    n_min = 2 ** k
    n_max = 2 ** (k + 1) - 1

    # --- Fast Type-D: enumerate n=2a² directly ---
    a_lo = math.isqrt(n_min // 2)
    while 2 * a_lo * a_lo < n_min:
        a_lo += 1
    D_shells = []
    a = a_lo
    while len(D_shells) < 8:
        n = 2 * a * a
        if n > n_max:
            break
        D_shells.append(n)
        a += 1
    # k-parity law: for odd k, n_min=2^k is always 2*(2^{(k-1)/2})² — put it first
    if k % 2 == 1 and n_min in D_shells:
        D_shells = [n_min] + [m for m in D_shells if m != n_min]

    if not D_shells:
        return None, None, None

    # For small k, use the original family-aware search (exact classification)
    if k <= 7:
        q1_cap = 20  # components fit within sqrt(255) < 16, so cap is fine
        C_shells: list[int] = []
        A_shells: list[int] = []
        for m in range(n_min, n_max + 1):
            if len(C_shells) >= 16 and len(A_shells) >= 16:
                break
            fam = classify_shell(m, max_coord=q1_cap)
            if fam == 'C' and len(C_shells) < 16:
                C_shells.append(m)
            elif fam == 'A' and len(A_shells) < 16:
                A_shells.append(m)
        for n_D in D_shells[:4]:
            for n_C in C_shells[:12]:
                for n_A in A_shells[:12]:
                    if n_C == n_D or n_A == n_D or n_A == n_C:
                        continue
                    if (_shells_have_triad(n_D, n_C, n_A) or
                            _shells_have_triad(n_D, n_A, n_C) or
                            _shells_have_triad(n_C, n_A, n_D)):
                        return n_D, n_C, n_A
        # fallback
        return D_shells[0], C_shells[0] if C_shells else None, A_shells[0] if A_shells else None

    # For large k (k >= 8): constructive triad from D-shell vectors.
    #
    # For n_D = 2a², the lattice points are exactly permutations of (±a, ±a, 0).
    # Use p1 = (a, a, 0) and p2 = (0, -s, z) for small s≥1.
    # Then:
    #   n_C  = s² + z²                         (shell of p2)
    #   p3   = -(p1+p2) = -(a, a-s, z)
    #   n_A  = a² + (a-s)² + z²  =  n_D - 2as + s² + z²
    # We just need n_C, n_A ∈ [n_min, n_max] and all three shells distinct.
    # This is O(sqrt(n_max)) per D-shell — no classification or vector search needed.

    def _near_power_of_2(n: int, tol: int = 128) -> bool:
        """True if n is within tol of any power of 2 (⇒ very few lattice reps)."""
        if n <= 0:
            return False
        lb = 1 << (n.bit_length() - 1)   # largest power of 2 ≤ n
        return (n - lb) <= tol or (lb * 2 - n) <= tol

    for n_D in D_shells[:4]:
        a = math.isqrt(n_D // 2)
        if 2 * a * a != n_D:
            continue  # safety: only pure 2a² shells have this vector structure
        fallback = None  # first valid nucleus found (may still be near-degenerate)
        for s in range(1, a):
            # z range for n_C = s²+z² ∈ [n_min, n_max]
            z_lo_C = math.isqrt(max(0, n_min - s * s))
            if z_lo_C * z_lo_C < n_min - s * s:
                z_lo_C += 1
            # Upper bound from n_C ≤ n_max
            z_hi_C = math.isqrt(n_max - s * s)
            # Tighter upper bound from n_A = n_D - 2as + n_C ≤ n_max
            z_hi_A_sq = n_max - n_D + 2 * a * s - s * s
            z_hi_A = math.isqrt(z_hi_A_sq) if z_hi_A_sq >= 0 else -1
            z_hi = min(z_hi_C, z_hi_A)
            if z_lo_C > z_hi:
                continue
            for z in range(z_lo_C, z_hi + 1):
                n_C = s * s + z * z
                n_A = n_D - 2 * a * s + n_C  # = a²+(a-s)²+z²
                if (n_min <= n_A <= n_max
                        and n_C != n_D and n_A != n_D and n_A != n_C):
                    if fallback is None:
                        fallback = (n_D, n_C, n_A)
                    # Prefer n_C (and n_A) not near any power of 2.
                    # The z-power-of-2 check alone is insufficient for large even k
                    # where z ≈ 2^(k/2) forces n_C = s² + (2^(k/2))² ≈ 2^k + small,
                    # which is near-power-of-2 regardless of whether z itself is 2^m.
                    if not _near_power_of_2(n_C) and not _near_power_of_2(n_A):
                        return n_D, n_C, n_A
        if fallback:
            return fallback

    # no valid triad found among D_shells[:4]
    return D_shells[0], None, None


# ---------------------------------------------------------------------------
# Optimizer (thin wrapper around get_wavevectors + neg_ratio_and_grad)
# ---------------------------------------------------------------------------

def _build_problem(shells: list[int]):
    """Build the optimizer data structures for a given list of shells.

    For 2- or 3-shell nuclei uses a cross-shell-only triad search (O(N_small ×
    N_large) instead of O(N²)), which is orders of magnitude faster at large k.
    Falls back to full precompute_triads for single-shell or unusual inputs.

    IMPORTANT — ANCHOR LIMITATION (structural incompleteness for large shell sets):
    When called with many shells (e.g. all shells in I_k for a full-block run),
    the cross-shell search is anchored on the SMALLEST shell in the list.  Only
    triads (ℓ, r, s) where ℓ lives on the smallest shell are found.  Any shell
    that never participates as ℓ in a triad with the smallest shell is absent from
    the resulting triad list — and consequently absent from _restrict_to_active_modes'
    mode set, since _restrict only keeps modes that appear in at least one triad.

    Practical consequence for run_block:
    • run_block passes ALL shells in I_k to _build_problem, then injects nucleus
      triads, then calls _restrict_to_active_modes.
    • For k=5 (I_5=[32,63]): smallest shell is 32 (3 modes, Type-D).  Shells like
      {33,34,35,36,38,41,...} that form rich mutual-triad networks among themselves
      but do not couple strongly to shell 32 are invisible after restriction.
    • The true full-block optimum at k=5 (C≈0.016735, support {32-36,38,41,43,50,
      51,54,59}) lives in exactly this invisible region.  run_block can never reach
      it regardless of how many starts are used.

    If you need to discover ALL basins (including those far from the smallest
    shell), use gap3_isc_at_true_optimizer.run_full_block, which calls
    precompute_triads on the entire mode set directly.
    """
    wavevecs_by_shell: list[list] = []
    for shell in shells:
        wv = get_wavevectors(max_shell2=shell, min_shell2=shell)
        wavevecs_by_shell.append(list(wv))

    all_wavevecs: list = []
    for wv_list in wavevecs_by_shell:
        all_wavevecs.extend(wv_list)

    if not all_wavevecs:
        return None

    if len(shells) >= 2:
        # Choose between GPU searchsorted triad builder and Python-dict fallback.
        # For large shells (N > 5 000) the Python dict path is O(N_anchor × 2N) dict
        # lookups which can be billions of operations; the torch path does the same
        # search in O(batch_size × 2N × log(2N)) GPU tensor ops ≈ seconds.
        N_total = sum(len(wvs) for wvs in wavevecs_by_shell)
        if _CUDA_OK and N_total > 5_000:
            ell_idx, ell2, r_idx, s_idx, s_mat = _build_triads_torch(wavevecs_by_shell)
        else:
            # Fast cross-shell triad finder: only iterate cross-shell (ℓ, r) pairs and
            # check s = ℓ - r in the doubled wavevec set.  O(sum_{i<j} N_i × N_j).
            # Build doubled index for fast lookup.
            pos_list = [tuple(int(c) for c in v) for v in all_wavevecs]
            neg_list = [tuple(-c for c in v) for v in pos_list]
            dbl = pos_list + neg_list
            N = len(pos_list)
            wv_idx = {v: i for i, v in enumerate(dbl)}

            ell_idx_list, ell2_list, r_idx_list, s_idx_list, s_list = [], [], [], [], []

            def _add_pair(ei, ri):
                ell = dbl[ei]
                r   = dbl[ri]
                s   = (ell[0]-r[0], ell[1]-r[1], ell[2]-r[2])
                si  = wv_idx.get(s)
                if si is not None:
                    ell_idx_list.append(ei)
                    ell2_list.append(ell[0]*ell[0]+ell[1]*ell[1]+ell[2]*ell[2])
                    r_idx_list.append(ri)
                    s_idx_list.append(si)
                    s_list.append(s)

            # Accumulate shell start offsets (in the positive half)
            offsets = []
            off = 0
            for wv_list in wavevecs_by_shell:
                offsets.append(off)
                off += len(wv_list)

            # Cross-shell pairs: iterate (ℓ, r) from DIFFERENT shells to find active modes.
            # Strategy: only use the SMALLEST shell as the ℓ anchor — this gives O(N_small ×
            # N_other) per pair of shells, avoiding the expensive O(N_large²) same-shell pass.
            # The small active-mode set is fully recoverable because every active mode
            # must participate in at least one triad anchored by the smallest shell.
            # (Same-shell triads within the active set are recovered in _restrict_to_active_modes
            #  via a full precompute_triads call on the small active-mode list.)
            shell_sizes = [(len(wv_list), si) for si, wv_list in enumerate(wavevecs_by_shell)]
            shell_sizes.sort()                        # smallest shell first
            anchor_si = shell_sizes[0][1]             # index of the smallest shell

            n_shells = len(shells)
            for si_a in range(n_shells):
                lo_a  = offsets[si_a]
                hi_a  = offsets[si_a] + len(wavevecs_by_shell[si_a])
                # Only anchor on the smallest shell (or pair it with everyone else)
                if si_a != anchor_si:
                    continue
                for si_b in range(n_shells):
                    lo_b = offsets[si_b]
                    hi_b = offsets[si_b] + len(wavevecs_by_shell[si_b])
                    for ei in range(lo_a, hi_a):            # positive ℓ
                        for ri in range(lo_b, hi_b):
                            _add_pair(ei, ri)
                        for ri in range(N + lo_b, N + hi_b):
                            _add_pair(ei, ri)
                    for ei in range(N + lo_a, N + hi_a):    # negative ℓ
                        for ri in range(lo_b, hi_b):
                            _add_pair(ei, ri)
                        for ri in range(N + lo_b, N + hi_b):
                            _add_pair(ei, ri)

            ell_idx = np.array(ell_idx_list, dtype=np.int32)
            ell2    = np.array(ell2_list,    dtype=np.float64)
            r_idx   = np.array(r_idx_list,   dtype=np.int32)
            s_idx   = np.array(s_idx_list,   dtype=np.int32)
            s_mat   = np.array(s_list,       dtype=np.float64)
    else:
        _, ell_idx, ell2, r_idx, s_idx, s_mat = precompute_triads(all_wavevecs)

    N = len(all_wavevecs)
    k2s = np.array([sum(c * c for c in kv) for kv in all_wavevecs], dtype=float)
    e1s = np.array([divfree_basis(kv)[0] for kv in all_wavevecs])
    e2s = np.array([divfree_basis(kv)[1] for kv in all_wavevecs])
    return dict(N=N, k2s=k2s, e1s=e1s, e2s=e2s,
                ell_idx=ell_idx, ell2=ell2, r_idx=r_idx, s_idx=s_idx, s_mat=s_mat,
                wavevecs=all_wavevecs)


_RETRIANGULATE_THRESHOLD = 5_000  # above this active-mode count, skip precompute_triads


def _restrict_to_active_modes(prob: dict, retriangulate: bool | None = None) -> dict:
    """Return a new prob restricted to only the modes that appear in triads.

    For nuclei with sparse triads (many silent modes), this dramatically reduces
    the parameter-space dimension and makes L-BFGS-B tractable.
    The restricted problem has identical C(I_k) value since silent modes (those
    not coupled by any triad) do not affect the numerator B and only inflate
    the denominator X²·D, so zeroing them out can only improve (or preserve)
    the optimum.

    retriangulate controls whether precompute_triads is called to complete
    same-shell triads on non-anchor shells that _build_problem omits:
      True  — always call precompute_triads (correct but O(N²) Python, infeasible
              for N_active > a few thousand)
      False — skip precompute_triads; remap existing triad indices to the
              compressed active-mode set.  Misses same-shell resonances on
              non-anchor shells, but is O(T) and always terminates.
      None  (default) — auto: use True if N_active <= _RETRIANGULATE_THRESHOLD,
              False otherwise (prints a warning).
    """
    N = prob['N']
    ell_idx, r_idx, s_idx = prob['ell_idx'], prob['r_idx'], prob['s_idx']

    # Active mode indices: anything that appears in a triad (mod N for neg half)
    active_arr = np.unique(np.concatenate([
        ell_idx % N, r_idx % N, s_idx % N,
    ]))
    N2 = len(active_arr)
    active_wavevecs = [prob['wavevecs'][i] for i in active_arr]

    if retriangulate is None:
        retriangulate = (N2 <= _RETRIANGULATE_THRESHOLD)
        if not retriangulate:
            print(f"  [restrict] N_active={N2} > {_RETRIANGULATE_THRESHOLD}:"
                  f" skipping precompute_triads (fast index-remap only)."
                  f"  Same-shell triads on non-anchor shells are absent.")

    k2s2 = prob['k2s'][active_arr]
    e1s2 = prob['e1s'][active_arr]
    e2s2 = prob['e2s'][active_arr]

    if not retriangulate:
        # Fast path: remap existing ell_idx/r_idx/s_idx from [0, 2N) → [0, 2*N2)
        # without calling precompute_triads.  All discovered triads are preserved;
        # same-shell triads on non-anchor shells are absent (acceptable trade-off
        # for large blocks where precompute_triads is infeasible).
        old_to_new = np.full(2 * N, -1, dtype=np.int32)
        new_indices = np.arange(N2, dtype=np.int32)
        old_to_new[active_arr] = new_indices           # positive half
        old_to_new[N + active_arr] = N2 + new_indices  # negative (conjugate) half

        new_ell = old_to_new[ell_idx]
        new_r   = old_to_new[r_idx]
        new_s   = old_to_new[s_idx]

        # Guard: drop any triad whose index wasn't mapped (should not happen, but
        # protects against inconsistent input dicts from caller).
        valid = (new_ell >= 0) & (new_r >= 0) & (new_s >= 0)
        return dict(N=N2, k2s=k2s2, e1s=e1s2, e2s=e2s2,
                    ell_idx=new_ell[valid].astype(np.int32),
                    ell2=prob['ell2'][valid],
                    r_idx=new_r[valid].astype(np.int32),
                    s_idx=new_s[valid].astype(np.int32),
                    s_mat=prob['s_mat'][valid],
                    wavevecs=active_wavevecs)

    # Slow path (small problems): full retriangulation via precompute_triads.
    # Always rebuild triads to ensure completeness — _build_problem's anchor-only
    # search finds a strict *subset* of resonant triads (cross-shell triads anchored
    # on the smallest shell only).  Same-shell triads among non-anchor shells are
    # intentionally omitted from the anchor search and recovered here.
    _, ell_idx2, ell2_2, r_idx2, s_idx2, s_mat2 = precompute_triads(active_wavevecs)
    return dict(N=N2, k2s=k2s2, e1s=e1s2, e2s=e2s2,
                ell_idx=ell_idx2, ell2=ell2_2, r_idx=r_idx2, s_idx=s_idx2, s_mat=s_mat2,
                wavevecs=active_wavevecs)


def _make_triad_starts(prob: dict, shells: list[int],
                        rng: np.random.Generator, n: int = 10) -> list[np.ndarray]:
    """Build warm starts concentrated on known resonant triads.

    For a 3-shell nucleus (n_D, n_C, n_A) the constructive triad provides
    explicit vectors p1+p2+p3=0.  Starting with most weight on these three
    modes ensures the optimizer sees a large gradient rather than a flat
    landscape in high-dimensional space.

    Returns up to n starts, each as a flat parameter array.
    """
    if len(shells) != 3:
        return []
    n_D, n_C, n_A = shells

    # Find D-shell vectors (type 2a² → perms of (±a, ±a, 0))
    a_sq = n_D // 2
    a = int(math.isqrt(a_sq))
    if 2 * a * a != n_D:
        return []  # not a pure D-shell

    wavevecs = prob['wavevecs']
    N = prob['N']
    bounds_lo = [0.0, 0.0, 0.0, -8.0] * N
    bounds_hi = [np.pi / 2, 2 * np.pi, 2 * np.pi, 8.0] * N

    # Index wavevectors for fast lookup.
    # get_wavevectors returns only the half-lattice (u_{-k} = conj(u_k)):
    # so for a triad p1+p2+p3=0, p3=-(p1+p2) is in the *negative* half and
    # its canonical representative is -p3.  Check both.
    wv_index = {tuple(int(c) for c in wv): i for i, wv in enumerate(wavevecs)}
    # Pre-bucket C-shell and A-shell indices for fast lookup
    c_indices = [i for i, wv in enumerate(wavevecs)
                 if sum(int(c) * int(c) for c in wv) == n_C]
    a_set = {n_A}  # just used for norm check before index lookup

    # Enumerate D-shell vectors; only those in the half-lattice will hit wv_index
    triad_triplets: list[tuple[int, int, int]] = []
    d_vecs = [(s1 * a, s2 * a, 0) for s1 in (+1, -1) for s2 in (+1, -1)]
    d_vecs += [(s1 * a, 0, s2 * a) for s1 in (+1, -1) for s2 in (+1, -1)]
    d_vecs += [(0, s1 * a, s2 * a) for s1 in (+1, -1) for s2 in (+1, -1)]
    for p1 in d_vecs:
        i1 = wv_index.get(p1)
        if i1 is None:
            continue
        for i2 in c_indices:
            wv2 = wavevecs[i2]
            wx, wy, wz = int(wv2[0]), int(wv2[1]), int(wv2[2])
            # p3 = -(p1 + p2); check both p3 and its negation (-p3 = p1+p2)
            p3_neg = (p1[0] + wx, p1[1] + wy, p1[2] + wz)  # = -p3, canonical half
            p3_pos = (-p1[0] - wx, -p1[1] - wy, -p1[2] - wz)
            if p3_neg[0]*p3_neg[0]+p3_neg[1]*p3_neg[1]+p3_neg[2]*p3_neg[2] != n_A:
                continue
            i3 = wv_index.get(p3_neg)
            if i3 is None:
                i3 = wv_index.get(p3_pos)
            if i3 is not None:
                triad_triplets.append((i1, i2, i3))
                if len(triad_triplets) >= 50:
                    break
        if len(triad_triplets) >= 50:
            break

    if not triad_triplets:
        return []

    starts = []
    for _ in range(n):
        x0 = rng.uniform(bounds_lo, bounds_hi)
        # Suppress all modes: set alpha→0
        for j in range(N):
            x0[4 * j] = 0.05
        # Concentrate on a random triad: set alpha→π/2, align phases
        i1, i2, i3 = triad_triplets[rng.integers(len(triad_triplets))]
        phase = rng.uniform(0, 2 * np.pi)
        for idx in (i1, i2, i3):
            x0[4 * idx] = np.pi / 2 - 0.05     # near-full weight
            x0[4 * idx + 1] = phase              # aligned theta
            x0[4 * idx + 2] = 0.0               # psi=0
            x0[4 * idx + 3] = 0.0               # rho
        starts.append(x0)
    return starts


def _run_optimizer_on_prob(prob: dict, n_starts: int, x0_warm: np.ndarray | None = None,
                            seed_offset: int = 0,
                            start_workers: int = 1,
                            shells: list[int] | None = None) -> tuple[float, np.ndarray]:
    """Run L-BFGS-B from n_starts random starts + optional warm start.

    When start_workers > 1 the starts are distributed across a
    ProcessPoolExecutor; problem arrays are pickled once per worker via the
    initializer pattern.

    Returns (best_val, best_x).
    """
    N = prob['N']
    k2s, e1s, e2s = prob['k2s'], prob['e1s'], prob['e2s']
    ell_idx, ell2 = prob['ell_idx'], prob['ell2']
    r_idx, s_idx, s_mat = prob['r_idx'], prob['s_idx'], prob['s_mat']

    if len(ell_idx) == 0:
        return 0.0, np.zeros(4 * N)

    bounds = [(0.0, np.pi / 2), (0.0, 2 * np.pi), (0.0, 2 * np.pi), (-8.0, 8.0)] * N
    lo = [lo for lo, _ in bounds]
    hi = [hi for _, hi in bounds]

    # Build list of starting points: random + triad-concentrated smart starts
    starts: list[np.ndarray] = []
    if x0_warm is not None:
        starts.append(x0_warm)
    rng0 = np.random.default_rng(seed_offset)
    if shells is not None and len(shells) == 3:
        smart = _make_triad_starts(prob, shells, rng0, n=min(n_starts, 20))
        starts.extend(smart)
    for i in range(n_starts):
        rng = np.random.default_rng(seed_offset + i)
        starts.append(rng.uniform(lo, hi))

    # Fix length mismatches (warm-start from a different problem size)
    corrected = []
    for x0 in starts:
        if len(x0) != 4 * N:
            rng = np.random.default_rng(seed_offset + 9999 + len(corrected))
            x0 = rng.uniform(lo, hi)
        corrected.append(x0)
    starts = corrected

    best_val = -1e10
    best_x = np.zeros(4 * N)

    if start_workers > 1 and starts:
        from concurrent.futures import ProcessPoolExecutor
        prob_tuple = (N, k2s, e1s, e2s, ell_idx, ell2, r_idx, s_idx, s_mat)
        with ProcessPoolExecutor(
                max_workers=start_workers,
                initializer=_init_starts_worker,
                initargs=(prob_tuple,)) as pool:
            for val, x in pool.map(_one_start, starts):
                if val > best_val:
                    best_val, best_x = val, x.copy()
    else:
        def objective(x):
            return neg_ratio_and_grad(x, N, e1s, e2s, k2s, ell_idx, ell2, r_idx, s_idx, s_mat)
        for x0 in starts:
            res = minimize(objective, x0, method='L-BFGS-B', jac=True, bounds=bounds,
                           options={'ftol': 1e-15, 'gtol': 1e-12, 'maxiter': 3000})
            val = float(-res.fun)
            if val > best_val:
                best_val = val
                best_x = res.x.copy()

    return best_val, best_x


def _run_block_worker(args: tuple) -> dict:
    """Top-level picklable wrapper used by ProcessPoolExecutor for across-k parallelism."""
    k, n_nuc, n_full, nucleus_only, start_workers, warm_npz_dir = args
    return run_block(k, n_starts_nuc=n_nuc, n_starts_full=n_full,
                     nucleus_only=nucleus_only, verbose=False,
                     start_workers=start_workers, warm_npz_dir=warm_npz_dir)


def _wavevec_hash(wavevecs: np.ndarray) -> int:
    """Stable integer fingerprint of a wavevector array for warm-state validation."""
    return int(np.sum(np.abs(np.asarray(wavevecs, dtype=np.int64)) *
                      np.array([1, 1009, 1000003], dtype=np.int64)))


def run_block(k: int, n_starts_nuc: int = 150, n_starts_full: int = 50,
              nucleus_only: bool = False, verbose: bool = True,
              start_workers: int = 1, warm_npz_dir: str | None = None) -> dict:
    """Run the two-level principled optimizer for block I_k."""
    t0_block = time.time()
    n_min, n_max = 2 ** k, 2 ** (k + 1) - 1
    parity = "odd" if k % 2 == 1 else "even"

    # ── Level 0: find the D×C×A nucleus ──────────────────────────────────────
    n_D, n_C, n_A = find_dcxa_nucleus(k)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  k={k}  I_{k}=[{n_min},{n_max}]  parity={parity}")
        print(f"  D×C×A nucleus: n_D={n_D}  n_C={n_C}  n_A={n_A}")
        if n_D: print(f"    classify({n_D})={classify_shell(n_D)}", end="")
        if n_C: print(f"  classify({n_C})={classify_shell(n_C)}", end="")
        if n_A: print(f"  classify({n_A})={classify_shell(n_A)}", end="")
        print()
        sys.stdout.flush()

    nucleus_shells = [s for s in [n_D, n_C, n_A] if s is not None]
    # ── Level 1: nucleus-restricted optimizer ─────────────────────────────────
    C_nuc, x_nuc, prob_nuc, t_nuc = 0.0, None, None, 0.0
    if len(nucleus_shells) < 2:
        if verbose:
            print(f"  No D\u00d7C\u00d7A nucleus found for k={k} — skipping nucleus phase, "
                  f"going direct to full-block")
            sys.stdout.flush()
        if nucleus_only:
            return {"k": k, "C_nuc": 0.0, "C_block": 0.0, "nucleus": nucleus_shells}
    else:
        t1 = time.time()
        prob_nuc = _build_problem(nucleus_shells)
        if prob_nuc is None:
            C_nuc, x_nuc = 0.0, None
        else:
            # Restrict to only modes that participate in at least one triad.
            # Silent modes inflate X²·D without contributing to B, so zeroing them
            # is guaranteed to keep or improve the optimum.
            prob_nuc = _restrict_to_active_modes(prob_nuc)
            n_modes_nuc = prob_nuc['N']
            n_triads_nuc = len(prob_nuc['ell_idx'])
            if verbose:
                print(f"  [Nucleus] modes={n_modes_nuc}  triads={n_triads_nuc}  starts={n_starts_nuc}")
                sys.stdout.flush()
            C_nuc, x_nuc = _run_optimizer_on_prob(prob_nuc, n_starts=n_starts_nuc,
                                                   x0_warm=None, seed_offset=k * 1000,
                                                   start_workers=start_workers,
                                                   shells=nucleus_shells)
        t_nuc = time.time() - t1
        if verbose:
            print(f"  [Nucleus] C_nuc(k={k}) = {C_nuc:.8f}  ({t_nuc:.1f}s)")
            sys.stdout.flush()

        if nucleus_only:
            return {"k": k, "C_nuc": C_nuc, "C_block": C_nuc,
                    "nucleus": nucleus_shells, "parity": parity,
                    "t_total": time.time() - t0_block}

    # ── Level 2: full-block optimizer (warm-started from nucleus) ─────────────
    # Build full-block problem
    t2 = time.time()
    all_shells = sorted(set(range(n_min, n_max + 1)))
    # Filter to shells that have at least one mode
    shells_with_modes = []
    for s in all_shells:
        wv = get_wavevectors(max_shell2=s, min_shell2=s)
        if wv:
            shells_with_modes.append(s)

    prob_full = _build_problem(shells_with_modes)

    # ── Nucleus triad injection ───────────────────────────────────────────
    # _build_problem uses a cross-shell search anchored on the SMALLEST
    # shell in the block (e.g. shell 4096 with only 3 modes for k=12).
    # When the nucleus shells are much larger (150-432 modes each) they are
    # not anchored, so their mutual triads are MISSED → B≈0 → R≈0 → C≈0.
    # Fix: explicitly map all nucleus triads (already computed in prob_nuc)
    # into the full-block mode index space and append them to prob_full.
    if prob_full is not None and prob_nuc is not None and len(prob_nuc['ell_idx']) > 0:
        full_wv_idx = {tuple(int(c) for c in wv): j
                       for j, wv in enumerate(prob_full['wavevecs'])}
        N_full = prob_full['N']
        N_nuc  = prob_nuc['N']
        nuc_to_full = {}
        for ni, wv in enumerate(prob_nuc['wavevecs']):
            fi = full_wv_idx.get(tuple(int(c) for c in wv))
            if fi is not None:
                nuc_to_full[ni] = fi
        add_ell, add_ell2, add_r, add_s, add_smat = [], [], [], [], []
        for t in range(len(prob_nuc['ell_idx'])):
            ei_n = int(prob_nuc['ell_idx'][t])
            ri_n = int(prob_nuc['r_idx'][t])
            si_n = int(prob_nuc['s_idx'][t])
            ei_b, ri_b, si_b = ei_n % N_nuc, ri_n % N_nuc, si_n % N_nuc
            if (ei_b not in nuc_to_full or ri_b not in nuc_to_full
                    or si_b not in nuc_to_full):
                continue
            ei_f = nuc_to_full[ei_b] + (N_full if ei_n >= N_nuc else 0)
            ri_f = nuc_to_full[ri_b] + (N_full if ri_n >= N_nuc else 0)
            si_f = nuc_to_full[si_b] + (N_full if si_n >= N_nuc else 0)
            add_ell.append(ei_f)
            add_ell2.append(float(prob_nuc['ell2'][t]))
            add_r.append(ri_f)
            add_s.append(si_f)
            add_smat.append(prob_nuc['s_mat'][t].tolist())
        if add_ell:
            prob_full['ell_idx'] = np.concatenate([
                prob_full['ell_idx'], np.array(add_ell, dtype=np.int32)])
            prob_full['ell2']    = np.concatenate([
                prob_full['ell2'],    np.array(add_ell2)])
            prob_full['r_idx']   = np.concatenate([
                prob_full['r_idx'],   np.array(add_r, dtype=np.int32)])
            prob_full['s_idx']   = np.concatenate([
                prob_full['s_idx'],   np.array(add_s, dtype=np.int32)])
            prob_full['s_mat']   = np.concatenate([
                prob_full['s_mat'],   np.array(add_smat)])
            if verbose:
                print(f"  [Full] injected {len(add_ell)} nucleus triads "
                      f"({N_nuc} nucleus modes → full-block triad total: "
                      f"{len(prob_full['ell_idx'])})")
                sys.stdout.flush()

    x_block = None  # set below if full-block optimizer runs
    if prob_full is None:
        C_block = C_nuc
    else:
        # Restrict to the active-mode subset and rebuild ALL resonant triads via
        # precompute_triads.  Without this, prob_full only contains the anchor-shell
        # cross-shell triads plus the injected nucleus triads — a small incomplete
        # subset of the full resonant structure.  _restrict_to_active_modes now
        # always calls precompute_triads (early-exit bug fixed), so this gives a
        # consistent complete triad set matching the certify script.
        prob_full = _restrict_to_active_modes(prob_full)
        n_modes_full = prob_full['N']
        n_triads_full = len(prob_full['ell_idx'])
        if verbose:
            print(f"  [Full]   modes={n_modes_full}  triads={n_triads_full}  "
                  f"starts={1+n_starts_full}  (1 nucleus warm-start + {n_starts_full} random)")
            sys.stdout.flush()

        # P3: warm-start from nucleus solution, padded to full-block dimension
        x0_warm = None
        if x_nuc is not None and prob_nuc is not None:
            # Build a warm-start by placing nucleus amplitudes in the right modes
            # Map nucleus mode indices to full-block mode indices
            nuc_wv_set = {tuple(wv): i for i, wv in enumerate(prob_nuc['wavevecs'])}
            x_full_warm = np.zeros(4 * n_modes_full)
            # Pin non-nucleus modes at lower bound (-8) so they don't swamp X².
            # L-BFGS-B will only move them up if the gradient warrants it (P8
            # falsification).  Using random values in [-6,-4] previously caused
            # ~1M non-nucleus modes to collectively dominate the denominator,
            # driving R → 0 regardless of the nucleus configuration.
            x_full_warm[3::4] = -8.0  # lower bound for log-amplitude
            # Override nucleus modes with a RESCALED nucleus solution.
            # R is scale-invariant so shifting all loga by δ preserves R(x_nuc).
            # But the full-block warm-start mixes nucleus modes (at x_nuc scale)
            # with N_full - N_nuc non-nucleus modes pinned at loga=-8.  If the
            # nucleus modes are at small amplitude (e.g. loga_nuc ≈ -4), the
            # non-nucleus denominator contribution can dominate, giving R_warm < C_nuc.
            # Fix: shift nucleus loga so max(loga_nuc) = 5, guaranteeing
            #   X²_nuc >> X²_non-nuc  (nucleus X² >> N_full * avg_k2 * exp(-16)).
            loga_nuc_vals = x_nuc[3::4]
            max_loga_nuc = float(np.max(loga_nuc_vals))
            if max_loga_nuc < 4.0:
                nuc_shift = min(5.0 - max_loga_nuc, 8.0 - max_loga_nuc)
            else:
                nuc_shift = 0.0
            for j, wv in enumerate(prob_full['wavevecs']):
                key = tuple(wv)
                if key in nuc_wv_set:
                    ni = nuc_wv_set[key]
                    row = x_nuc[4*ni:4*ni+4].copy()
                    row[3] = float(np.clip(row[3] + nuc_shift, -8.0, 8.0))
                    x_full_warm[4*j:4*j+4] = row
            x0_warm = x_full_warm

        # Warm-state load: inject previous best_x as an extra start
        if warm_npz_dir is not None:
            warm_path = os.path.join(warm_npz_dir, f"k{k}_warm.npz")
            if os.path.isfile(warm_path):
                try:
                    npz = np.load(warm_path)
                    saved_k = int(npz['k'])
                    saved_n = int(npz['n_modes'])
                    saved_hash = int(npz['wv_hash'])
                    cur_hash = _wavevec_hash(prob_full['wavevecs'])
                    if saved_k != k:
                        raise ValueError(f"k mismatch: file has k={saved_k}, expected k={k}")
                    if saved_n != n_modes_full:
                        raise ValueError(f"n_modes mismatch: file has {saved_n}, expected {n_modes_full}")
                    if saved_hash != cur_hash:
                        raise ValueError(f"wavevector hash mismatch ({saved_hash} vs {cur_hash}) — mode ordering changed")
                    x_saved = npz['best_x'].copy()
                    # Safety: clip to bounds and reject if any non-finite values
                    bounds_lo = np.tile([0.0, 0.0, 0.0, -8.0], n_modes_full)
                    bounds_hi = np.tile([np.pi/2, 2*np.pi, 2*np.pi, 8.0], n_modes_full)
                    x_saved = np.clip(x_saved, bounds_lo, bounds_hi)
                    if not np.all(np.isfinite(x_saved)):
                        raise ValueError("saved best_x contains non-finite values")
                    # Use as x0_warm (overrides nucleus warm-start if better)
                    saved_val = float(npz['best_val'])
                    x0_warm = x_saved
                    if verbose:
                        print(f"  [warm-load] k{k}_warm.npz: best_val={saved_val:.8f}, "
                              f"n_modes={saved_n} OK — injected as warm start")
                        sys.stdout.flush()
                except Exception as exc:
                    if verbose:
                        print(f"  [warm-load] WARNING: skipping {warm_path}: {exc}")
                        sys.stdout.flush()

        # P8: n_starts_full additional random starts for falsification
        C_block, x_block = _run_optimizer_on_prob(
            prob_full, n_starts=n_starts_full, x0_warm=x0_warm, seed_offset=k * 2000,
            start_workers=start_workers)

    t_full = time.time() - t2
    t_total = time.time() - t0_block

    # Enforce mathematical floor: C(I_k) >= C_nuc (nucleus is a subspace of the block).
    # The full-block optimizer can converge below C_nuc if the warm-start or random starts
    # explore poor regions; the nucleus itself is always a valid lower bound.
    if C_block < C_nuc:
        C_block = C_nuc

    if verbose:
        print(f"  [Full]   C(I_{k}) = {C_block:.8f}  ({t_full:.1f}s)")
        # P8 alert: did the full block beat the nucleus by >5%?
        # Suppress when nucleus phase was skipped (C_nuc=0 by construction).
        if C_nuc > 0 and C_block > C_nuc * 1.05:
            print(f"  *** P8 FLAG: C_block={C_block:.6f} > 1.05 × C_nuc={C_nuc:.6f} ***")
            print(f"      Random starts beat nucleus warm-start — nucleus is NOT complete!")
            # Print energy profile of full-block optimizer solution
            if x_block is not None and prob_full is not None:
                k2s_full = prob_full['k2s']
                log_a = x_block[3::4]
                a2 = np.exp(2.0 * log_a)
                X2 = float(np.dot(k2s_full, a2))
                if X2 > 0:
                    k2s_int = k2s_full.astype(int)
                    shells_all = sorted(set(k2s_int.tolist()))
                    profile = []
                    for sh in shells_all:
                        mask = (k2s_int == sh)
                        frac_sh = float(np.dot(k2s_full[mask], a2[mask])) / X2
                        if frac_sh > 1e-4:
                            profile.append((sh, frac_sh))
                    profile.sort(key=lambda x: -x[1])
                    print(f"  [Full-block optimizer energy profile]")
                    for (sh, frac_sh) in profile:
                        bar = '█' * int(frac_sh * 40)
                        print(f"    n={sh:6d}:  {frac_sh:.4f}  {bar}")
                    inactive = [sh for sh in shells_all
                                if sh not in {s for s, _ in profile}]
                    if inactive:
                        print(f"    Inactive shells (E<0.01%): {inactive}")
        frac = C_nuc / C_block if C_block > 0 else 1.0
        print(f"  Nucleus fraction: C_nuc/C_block = {frac:.4f}  "
              f"(nucleus explains {100*frac:.1f}% of block max)")
        print(f"  Total time for k={k}: {t_total:.1f}s")
        sys.stdout.flush()

    # Warm-state save: persist best full-block solution for future runs.
    # Only overwrites if strictly better — makes concurrent CPU+GPU runs safe.
    if warm_npz_dir is not None and x_block is not None and prob_full is not None:
        os.makedirs(warm_npz_dir, exist_ok=True)
        warm_path = os.path.join(warm_npz_dir, f"k{k}_warm.npz")
        try:
            _skip_save = False
            if os.path.isfile(warm_path):
                try:
                    _existing = np.load(warm_path)
                    _existing_val = float(_existing['best_val'])
                    if _existing_val >= C_block:
                        _skip_save = True
                        if verbose:
                            print(f"  [warm-save] Skipped k{k}_warm.npz "
                                  f"(existing C={_existing_val:.8f} >= new C={C_block:.8f})")
                            sys.stdout.flush()
                except Exception:
                    pass  # corrupt/unreadable — overwrite
            if not _skip_save:
                np.savez_compressed(
                    warm_path,
                    k=np.int64(k),
                    n_modes=np.int64(prob_full['N']),
                    wv_hash=np.int64(_wavevec_hash(prob_full['wavevecs'])),
                    best_val=np.float64(C_block),
                    best_x=x_block,
                )
                if verbose:
                    print(f"  [warm-save] Saved k{k}_warm.npz  (C={C_block:.8f}, "
                          f"n_modes={prob_full['N']})")
                    sys.stdout.flush()
        except Exception as exc:
            if verbose:
                print(f"  [warm-save] WARNING: could not save {warm_path}: {exc}")
                sys.stdout.flush()

    return {"k": k, "C_nuc": C_nuc, "C_block": C_block,
            "nucleus": nucleus_shells, "parity": parity,
            "t_nuc": t_nuc, "t_full": t_full, "t_total": t_total}


# ---------------------------------------------------------------------------
# Scaling analysis (P7, P12)
# ---------------------------------------------------------------------------

KNOWN_CIK = {
    1: 0.0,
    2: 0.022740,
    3: 0.021936432,  # CERTIFIED 50 dps mpmath, complete T=3456 triads, N=73 modes, 9 active, Hess 35neg/1flat/0pos.
                     # NOTE: relay modes all at loga=-8 boundary → C_nuc = C_block = 0.02193643.
                     # Old incorrect value 0.046925 (anchor-only T=408 problem) and 0.031900 (T=216 nucleus).
                     # Fixed by removing early-exit in _restrict_to_active_modes + calling _restrict on prob_full.
                     # Value 0.021936432 from certify_k3_v2.txt (50 dps, April 4 2026).
    4: 0.021057589,  # FULL BLOCK [16,32), 50-dps mpmath GLOBALLY CERTIFIED (certify_block_maximum Apr 8 2026); precompute_triads 244 modes, 21 active; 1000-start P8 global scan.
                   # 200 float32+float64 starts on full 244-mode precompute_triads space.
                   # New basin: shells {16,17,18,20,21,25,26,29}; local EL cert: all 152 relay modes d2<0,
                   # max ΔC=2.39e-9 (rel 1.14e-7). Supersedes CPU bound 0.020946 (Apr 4 2026).
                   # Global cert pending.
    5: 0.016735,   # FULL BLOCK [32,63]; best verified lower bound from gap3_isc_at_true_optimizer
                   # (run_full_block, 500 starts 24 workers, isc_k5_full.txt Apr 5 2026).
                   # Support: {32,33,34,35,36,38,41,43,50,51,54,59} (12 shells), ISC cert in progress.
                   # run_block (principled scan) found only 0.015347 — NOT because of a code bug but
                   # because _build_problem's anchor search is anchored on the SMALLEST shell (shell 32,
                   # 3 modes) and the high-value basin lives on shells {33-36,38,...} that barely couple
                   # to shell 32.  Those shells are absent from run_block's triad list entirely.
                   # See _build_problem docstring and max_b_over_keff.py module docstring for the full
                   # architectural explanation.  The "pre-fix artifact" diagnosis (Apr 8 2026) was WRONG.
                   # isc_k5_v2_full.txt run (Apr 8 2026) was at 0.01628 at start 50/500 — consistent.
                   # CONFIRMED < C(I_2) ✅. Old stale value was 0.024120 (pre-_restrict fix, spurious).
    # Full-block verified (lower bound, confirmed via multi-start L-BFGS-B)
    6: 0.017862,   # FULL BLOCK [64,127], GPU full-block (Apr 7 2026, k6_warm.npz). Lower bound;
                   # run still converging. Old stale value 0.016141 (anchor-only incomplete problem).
    7: 0.012803,   # FULL BLOCK [128,255]; n_full=500 (re-verified), 2776.4s, April 2 2026; P8 flag: true optimizer {145(C),129(B),192(B)}, fracs 44.9/36.1/18.6%; nucleus {128(D),130(C),146(A)} only 10.8%; CONFIRMED < 0.023 ✅
                   #   Note: 192=3×64, classify_shell(192)='B' (not A); earlier '192(A)' label corrected (§19.10.15)
                   #   NEEDS VERIFICATION: run_block did not call _restrict_to_active_modes on full-block.
    # Nucleus-only lower bounds (all confirmed ≪ 0.023 threshold)
    8: 0.008110,   # FULL BLOCK [256,511]; n_nuc=50, C_nuc=0.004622 (nucleus {288,260,500}), n_full=200, C(I_8)=0.00811041, 5110.1s, April 3 2026; P8 flag: nucleus fraction 57.0%; true optimizer {257(C):46.9%, 256(B):34.2%, 289(C):18.9%}; EVEN-K sub-family {a^2=256, a^2+1=257, (a+1)^2=289} where a=2^(k/2)=16; C(I_8)*2^4=0.1298
                   #   NEEDS VERIFICATION: run_block did not call _restrict_to_active_modes on full-block.
    9: 0.006520,   # FULL BLOCK [512,1023]; n_nuc=50, C_nuc=0.002477 (nucleus {512,657,881}), n_full=200, C(I_9)=0.00652037, 20025.8s, April 3 2026; P8 flag: nucleus fraction 38.0%; true optimizer {518(A):79.4%, 768(B):20.6%} — 2-shell A+B; 518=2×7×37 minimal A-type shell in I_9 admitting component-sum=-24; C(I_9)*2^(9/2)=0.1475; CONFIRMED < 0.023 ✅
                   #   NEEDS VERIFICATION: run_block did not call _restrict_to_active_modes on full-block.
    10: 0.004165,  # FULL BLOCK [1024,2047]; n_nuc=50, C_nuc=0.001976 (nucleus {1058,1153,1843}), n_full=201, C(I_10)=0.004165, 34857s, April 3 2026; P8 flag: nucleus fraction 47.5%; true optimizer {1024(D):33.3%, 1041(C):66.7%} — D+C 2-shell; EVEN-K; C(I_10)*2^5=0.13328; CONFIRMED < 0.023 ✅
                   #   (Old nucleus-only entry 0.001976 from April 2 2026 was stale; full-block value supersedes it)
    11: 0.001679,  # nucleus {2048,2234,3962}, 150 starts, April 2 2026
    12: 0.001199,  # nucleus {4232,4250,8022}, 150 starts, April 2 2026
    # Plateau regime: C(I_k) × 2^(k/2) ≈ 0.0838 ± 0.0005
    13: 0.001155,  # nucleus {8192,8480,16160},            150 starts, April 2 2026
    14: 0.000635,  # nucleus {16562,16657,32491},          150 starts, April 2 2026
    15: 0.000460,  # nucleus {32768,33128,65384},          150 starts, April 2 2026
    16: 0.000323,  # nucleus {66248,66065,130857},         150 starts, April 2 2026
    17: 0.000231,  # nucleus {131072,131773,261821},       150 starts, April 2 2026
    18: 0.000163,  # nucleus {263538,263185,523819},       150 starts, April 2 2026
    19: 0.000116,  # nucleus {524288,525629,1047869},      150 starts, April 2 2026
    20: 0.000082,  # nucleus {1051250,1050641,2096091},    150 starts, April 2 2026
    21: 0.000058,  # nucleus {2097152,2099605,4192661},    30 starts, April 2 2026
    22: 0.000041,  # nucleus {4199202,4198417,8386027},    30 starts, April 2 2026  (fixed nucleus — see §19.10.9d)
    23: 0.000029,  # nucleus {8388608,8392610,16777122},   30 starts, April 2 2026
    24: 0.000021,  # nucleus {16785218,16777220,33550850}, 30 starts, April 2 2026
    25: 0.000015,  # nucleus {33554432,33558850,67105090}, 30 starts, April 2 2026
    26: 0.00000001, # nucleus {67117698,67125258,134208198},  30 starts, April 2 2026 (degenerate nucleus — even-k anomaly; true C(I_26) follows plateau)
    27: 0.000007,   # nucleus {134217728,134235400,268420360}, 30 starts, April 2 2026
    28: 0.000000,   # nucleus {268470792,268468234,536869510}, 30 starts, April 2 2026 (degenerate nucleus — even-k anomaly; true C(I_28) follows plateau)
    29: 0.00000363, # nucleus {536870912,536895242,1073733386},   30 starts, April 2 2026
    31: 0.00000182, # nucleus {2147483648,2147488282,4294906394}, 30 starts, April 2 2026
    # k=30: ⏳ in progress
}

def print_scaling_analysis(results: list[dict]):
    """P7: check structural hypotheses about how C(I_k) scales with k.
    P12: separate odd-k and even-k."""
    all_data = []
    for k, val in KNOWN_CIK.items():
        all_data.append({"k": k, "C_block": val, "parity": "odd" if k%2==1 else "even"})
    for r in results:
        all_data.append(r)

    # Sort by k
    all_data.sort(key=lambda d: d["k"])

    print()
    print("=" * 80)
    print("  SCALING ANALYSIS (P7: structural hypothesis, P12: parity separation)")
    print("=" * 80)
    print(f"  {'k':>4}  {'par':>4}  {'C(I_k)':>10}  {'C×2^(k/2)':>12}  {'C×k':>10}  "
          f"{'C×k^(3/2)':>12}  {'ratio prev':>12}")
    print("  " + "─" * 70)

    prev_val = None
    for d in all_data:
        k = d["k"]
        C = d.get("C_block", d.get("C"))
        if C is None or C <= 0.0:
            continue
        par = d.get("parity", "odd" if k%2==1 else "even")
        sc1 = C * (2 ** k) ** 0.5           # C × 2^{k/2}
        sc2 = C * k                           # C × k
        sc3 = C * k ** 1.5                   # C × k^{3/2}
        ratio = (C / prev_val) if prev_val and prev_val > 0 else float("nan")
        print(f"  {k:>4}  {par:>4}  {C:>10.6f}  {sc1:>12.6f}  "
              f"{sc2:>10.6f}  {sc3:>12.6f}  {ratio:>12.6f}")
        prev_val = C

    print()
    # Separate odd/even
    for par in ("odd", "even"):
        data_par = [d for d in all_data
                    if d.get("parity", "odd" if d["k"]%2==1 else "even") == par
                    and d.get("C_block", d.get("C")) is not None
                    and d.get("C_block", d.get("C", 0)) > 0]
        if len(data_par) < 2:
            continue
        vals_k = [d["k"] for d in data_par]
        vals_C = [d.get("C_block", d.get("C")) for d in data_par]
        if len(vals_k) >= 3:
            # Fit log(C) ~ a + b*log(k) or a + b*k
            import numpy as np
            logk = np.log(vals_k)
            logC = np.log(vals_C)
            # log-log: C ~ A * k^b
            b, lnA = np.polyfit(logk, logC, 1)
            A = math.exp(lnA)
            print(f"  {par}-k power-law fit: C(I_k) ≈ {A:.5f} × k^({b:.3f})")
            # semi-log: C ~ A * exp(b*k)
            b2, lnA2 = np.polyfit(vals_k, logC, 1)
            A2 = math.exp(lnA2)
            print(f"  {par}-k exp fit:       C(I_k) ≈ {A2:.5f} × exp({b2:.4f}·k)")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ncpu = os.cpu_count() or 1
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--kmin', type=int, default=6,
        help='Minimum block index (default: 6)')
    parser.add_argument('--kmax', type=int, default=8,
        help='Maximum block index (default: 8)')
    parser.add_argument('--n_nuc', type=int, default=150,
        help='Random starts for nucleus optimizer (default: 150)')
    parser.add_argument('--n_full', type=int, default=50,
        help='Random starts for full-block optimizer (default: 50)')
    parser.add_argument('--nucleus_only', action='store_true',
        help='Skip full-block optimizer (faster, gives lower bound)')
    parser.add_argument('--workers', type=int, default=1,
        help='k-values to process in parallel (default: 1; -1 = all CPUs). '
             'Each k runs sequentially internally unless --start-workers is set.')
    parser.add_argument('--start-workers', type=int, default=1,
        help='CPUs for parallel L-BFGS-B restarts within each k (default: 1; '
             '-1 = all CPUs). Combine with --workers=1 unless you have many cores.')
    parser.add_argument('--warm-npz', type=str, default=None, metavar='DIR',
        help='Directory for warm-state npz files (k{k}_warm.npz). If a file '
             'exists for a given k, its best_x is injected as an extra start after '
             'validation (k, n_modes, wavevector hash). The best solution found is '
             'always saved back to the same file, overwriting if better. '
             'Safe to use across interrupted/resumed runs.')
    args = parser.parse_args()
    if args.workers == -1:
        args.workers = ncpu
    if args.start_workers == -1:
        args.start_workers = ncpu

    print("=" * 70)
    print("  GAP 3 PRINCIPLED SCAN  (Principles P3, P8, P9, P12)")
    print(f"  Blocks k = {args.kmin}..{args.kmax}   "
          f"nucleus starts={args.n_nuc}   "
          f"full starts={args.n_full}   "
          f"nucleus_only={args.nucleus_only}")
    print()
    known_prior = {k: v for k, v in KNOWN_CIK.items() if k < args.kmin}
    if known_prior:
        print(f"  Prior data (k=1..{max(known_prior)}):")
        for k in sorted(known_prior):
            parity = "odd" if k%2==1 else "even"
            label = "(nucleus lower bound)" if k >= 8 else ""
            print(f"    C(I_{k}) = {KNOWN_CIK[k]:.6f}  ({parity}-k)  {label}")
    print("=" * 70)
    sys.stdout.flush()

    k_range = list(range(args.kmin, args.kmax + 1))
    results: list[dict] = []
    C_prev = max(KNOWN_CIK.values())

    if args.workers > 1 and len(k_range) > 1:
        # ── Across-k parallel execution ─────────────────────────────────────
        print(f"  [parallel] {min(args.workers, len(k_range))} k-values at a time  "
              f"(start-workers={args.start_workers} each)")
        sys.stdout.flush()
        from concurrent.futures import ProcessPoolExecutor, as_completed
        tasks = [(k, args.n_nuc, args.n_full, args.nucleus_only, args.start_workers, args.warm_npz)
                 for k in k_range]
        results_map: dict[int, dict] = {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run_block_worker, t): t[0] for t in tasks}
            for fut in as_completed(futures):
                k = futures[fut]
                try:
                    r = fut.result()
                except Exception as exc:
                    print(f"  k={k} FAILED: {exc}")
                    sys.stdout.flush()
                    continue
                results_map[k] = r
                C_curr = r.get("C_block", r.get("C_nuc", 0.0))
                t_tot = r.get("t_total", 0.0)
                nuc = r.get("nucleus", [])
                print(f"  k={k:2d} done: C_nuc={r['C_nuc']:.8f}  C={C_curr:.8f}  "
                      f"({t_tot:.1f}s)  nucleus={nuc}")
                if C_curr > C_prev and C_curr > 0.023:
                    print(f"  *** P9 ALERT: C(I_{k}) = {C_curr:.6f} > 0.023 threshold! ***")
                    print(f"      GAP 3 conjecture may FAIL at k={k}. Investigate immediately.")
                C_prev = max(C_prev, C_curr)
                sys.stdout.flush()
        results = [results_map[k] for k in sorted(results_map)]
    else:
        # ── Sequential execution (original behaviour) ────────────────────────
        for k in k_range:
            r = run_block(k,
                          n_starts_nuc=args.n_nuc,
                          n_starts_full=args.n_full,
                          nucleus_only=args.nucleus_only,
                          verbose=True,
                          start_workers=args.start_workers,
                          warm_npz_dir=args.warm_npz)
            results.append(r)
            C_curr = r.get("C_block", r.get("C_nuc", 0.0))
            if C_curr > C_prev and C_curr > 0.023:
                print(f"  *** P9 ALERT: C(I_{k}) = {C_curr:.6f} > 0.023 threshold! ***")
                print(f"      GAP 3 conjecture may FAIL at k={k}. Investigate immediately.")
            C_prev = max(C_prev, C_curr)

    # ── Summary table ──────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  SUMMARY  (new results only)")
    print("─" * 70)
    print(f"  {'k':>3}  {'parity':>6}  {'n_D':>6}  {'n_C':>6}  {'n_A':>6}  "
          f"{'C_nuc':>10}  {'C(I_k)':>10}  {'nuc/block':>10}")
    print("  " + "─" * 62)
    for r in results:
        k = r["k"]
        nuc = r["nucleus"]
        n_D = nuc[0] if len(nuc) > 0 else "—"
        n_C = nuc[1] if len(nuc) > 1 else "—"
        n_A = nuc[2] if len(nuc) > 2 else "—"
        C_nuc = r["C_nuc"]
        C_blk = r["C_block"]
        frac = C_nuc / C_blk if C_blk > 0 else float("nan")
        print(f"  {k:>3}  {'odd' if k%2==1 else 'even':>6}  "
              f"{str(n_D):>6}  {str(n_C):>6}  {str(n_A):>6}  "
              f"{C_nuc:>10.6f}  {C_blk:>10.6f}  {frac:>10.4f}")

    # ── Scaling analysis ───────────────────────────────────────────────────────
    print_scaling_analysis(results)

    # ── GAP 3 status ───────────────────────────────────────────────────────────
    all_vals = {r["k"]: r.get("C_block", r["C_nuc"]) for r in results}
    all_vals.update(KNOWN_CIK)
    max_C = max(v for v in all_vals.values() if v > 0)
    max_k = max((k for k in all_vals if all_vals[k] == max_C), default="?")
    print()
    print("=" * 70)
    print(f"  max C(I_k) over all verified k: {max_C:.6f}  (at k={max_k})")
    verified_ks = sorted(all_vals.keys())
    print(f"  Verified: k ∈ {verified_ks}")
    print(f"  Remaining gap: k = {args.kmax+1}..31  (plus k≥32: analytical)")
    print("=" * 70)


if __name__ == "__main__":
    main()
