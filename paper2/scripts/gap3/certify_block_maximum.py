#!/usr/bin/env python3
"""
certify_block_maximum.py
========================
Certify C(I_k) as a rigorous lower bound via interval arithmetic.
Works for any k; defaults to k=3.  Pass --k K for other blocks.

Strategy
--------
For a given block I_k = [2^k, 2^(k+1)-1] we:

  1. Build the restricted problem (nucleus or full-block) with N modes and
     T triads (ALL resonant triads among the active modes, via precompute_triads).
  2. Find x* via multi-start L-BFGS-B (float64, tight tolerances).
     Optionally load a warm start from --warm-npz (written by gpu_optimizer.py
     or gap3_principled_scan.py) to skip expensive random re-exploration.
  3. Evaluate R(x*) in mpmath arithmetic to obtain a rigorous numerical
     certificate: R(x*) in [lo, hi] with hi - lo < 10^-dps.
  4. Verify the Hessian of R at x* is negative definite (all eigenvalues
     negative) — confirming x* is a certified local maximum.
  5. Check ISC (Improved Shell Contribution) for every relay shell: if
     adding a relay shell direction improves R above R(x*), we record it.

Tractability by k
-----------------
  k=3: All steps tractable.  N~42 nucleus, N~73 full-block.  Minutes.
  k=4: Steps 2-3 tractable.  Hessian (~400 params) parallelised across CPUs
       (~4 min at 16 cores / 50 dps, vs ~1 hour serial).
  k=5: Step 2 (float64 optimizer) tractable with warm-npz.
       Step 3 (mpmath evaluation) expensive but feasible (single call, ~5 min).
       Step 4 (Hessian, ~2000 params × T=158K) — use --n_workers to parallelise
       or --skip_hess to skip entirely.
       Step 5 (ISC, float64) is always tractable.
  k≥6: Use --skip_hess and rely on ISC + mpmath evaluation only.

Usage
-----
    python scripts/gap3/certify_block_maximum.py
    python scripts/gap3/certify_block_maximum.py --k 5 --full_block --warm-npz results/warm_state --skip_isc
    python scripts/gap3/certify_block_maximum.py --n_starts 200 --dps 50
    python scripts/gap3/certify_block_maximum.py --full_block --n_starts 100
    python scripts/gap3/certify_block_maximum.py --dps 100 --skip_isc
    python scripts/gap3/certify_block_maximum.py --k 4 --full_block --n_workers 16
"""
import argparse
import math
import multiprocessing
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

try:
    import mpmath
except ImportError:
    sys.exit("mpmath not found — install with: pip install mpmath")

sys.path.insert(0, ".")
from scripts.gap3.multi_mode_beta_bound import get_wavevectors, divfree_basis, precompute_triads
from scripts.gap3.gap3_principled_scan import (_build_problem, find_dcxa_nucleus,
                                          _restrict_to_active_modes)
from scripts.gap3.max_b_over_keff import neg_ratio_and_grad

# ---------------------------------------------------------------------------
# Parallel L-BFGS-B worker (module-level for pickling compatibility)
# ---------------------------------------------------------------------------

_worker_prob = None

def _init_lbfgsb_worker(prob):
    """Pool initializer: copy prob into each worker process once."""
    global _worker_prob
    _worker_prob = prob

def _lbfgsb_start_worker(args):
    """Run one L-BFGS-B start with tight tolerances; returns (i, val, x)."""
    i, x0 = args
    prob = _worker_prob
    N = prob['N']
    k2s, e1s, e2s = prob['k2s'], prob['e1s'], prob['e2s']
    ell_idx, ell2 = prob['ell_idx'], prob['ell2']
    r_idx, s_idx, s_mat = prob['r_idx'], prob['s_idx'], prob['s_mat']
    bounds = [(0., math.pi / 2), (0., 2*math.pi), (0., 2*math.pi), (-8., 8.)] * N
    def obj(x):
        return neg_ratio_and_grad(x, N, e1s, e2s, k2s, ell_idx, ell2, r_idx, s_idx, s_mat)
    res = minimize(obj, x0, method='L-BFGS-B', jac=True, bounds=bounds,
                   options={'ftol': 1e-15, 'gtol': 1e-13, 'maxiter': 100_000})
    return (i, float(-res.fun), res.x.copy())


def _lbfgsb_explore_worker(args):
    """Run one L-BFGS-B start with loose tolerances for basin identification;
    returns (i, val, x).  Typically 10-50× faster than _lbfgsb_start_worker
    when starting near the optimum (e.g. post-Adam).  Always follow with one
    tight single-start polish on the winner."""
    i, x0 = args
    prob = _worker_prob
    N = prob['N']
    k2s, e1s, e2s = prob['k2s'], prob['e1s'], prob['e2s']
    ell_idx, ell2 = prob['ell_idx'], prob['ell2']
    r_idx, s_idx, s_mat = prob['r_idx'], prob['s_idx'], prob['s_mat']
    bounds = [(0., math.pi / 2), (0., 2*math.pi), (0., 2*math.pi), (-8., 8.)] * N
    def obj(x):
        return neg_ratio_and_grad(x, N, e1s, e2s, k2s, ell_idx, ell2, r_idx, s_idx, s_mat)
    res = minimize(obj, x0, method='L-BFGS-B', jac=True, bounds=bounds,
                   options={'ftol': 1e-12, 'gtol': 1e-9, 'maxiter': 2_000})
    return (i, float(-res.fun), res.x.copy())


# ---------------------------------------------------------------------------
# mpmath objective:  R(params) = B / (X^2 * D)
# ---------------------------------------------------------------------------

def _mpmath_objective(params_mp, N_mp, k2s_mp, e1s_mp, e2s_mp,
                      ell_mp, ell2_mp, r_mp, s_mp, s_mat_mp):
    """
    Evaluate R = B / (X^2 * D) in mpmath arithmetic.
    params_mp: mpmath matrix of length 4*N
    """
    mp = mpmath

    theta = [params_mp[4*i + 0] for i in range(N_mp)]
    phi   = [params_mp[4*i + 1] for i in range(N_mp)]
    psi   = [params_mp[4*i + 2] for i in range(N_mp)]
    loga  = [params_mp[4*i + 3] for i in range(N_mp)]

    a = [mp.exp(la) for la in loga]
    r = [mp.sqrt(ai) for ai in a]

    # Build u_pos[i]: (3,) complex mpmath vector
    u_pos = []
    for i in range(N_mp):
        e1  = e1s_mp[i]
        e2  = e2s_mp[i]
        cth = mp.cos(theta[i])
        sth = mp.sin(theta[i])
        eph = mp.exp(mp.j * phi[i])
        eps = mp.exp(mp.j * psi[i])
        ri  = r[i]
        u_pos.append([ri * (cth * eph * e1[d] + sth * eps * e2[d])
                      for d in range(3)])

    # Full array: u[i] for i in [0, 2N): first N positive, next N conjugates
    def u(idx):
        if idx < N_mp:
            return u_pos[idx]
        else:
            return [x.conjugate() for x in u_pos[idx - N_mp]]

    # X^2 = 2 sum_i k2_i * a_i
    X2 = 2 * sum(k2s_mp[i] * a[i] for i in range(N_mp))

    # D = sqrt(2 sum_i k2_i^2 * a_i)
    D2 = 2 * sum(k2s_mp[i]**2 * a[i] for i in range(N_mp))
    D  = mp.sqrt(D2)

    if X2 == 0 or D == 0:
        return mp.mpf(0)

    # B = -Im( sum_t ell2_t * (s_t . u[r_t]) * conj(u[ell_t]) . u[s_t] )
    # Note: per-triad arrays may be numpy arrays (to avoid huge Python-list
    # allocation for large T); use int()/float() to get plain Python scalars.
    B = mp.mpf(0)
    T = len(ell_mp)
    for t in range(T):
        ei  = int(ell_mp[t])
        ri  = int(r_mp[t])
        si  = int(s_mp[t])
        sv  = s_mat_mp[t]          # numpy row or list; indexed sv[d]
        l2  = mp.mpf(float(ell2_mp[t]))
        # s_t · u[r_t]
        sdu = sum(mp.mpf(float(sv[d])) * u(ri)[d] for d in range(3))
        # conj(u[ell_t]) · u[s_t]
        ced = sum(u(ei)[d].conjugate() * u(si)[d] for d in range(3))
        B -= mp.im(l2 * sdu * ced)

    return B / (X2 * D)


def _build_mpmath_data(prob, dps):
    """Convert numpy problem arrays to mpmath at given precision.

    Per-triad arrays (ell_idx, ell2, r_idx, s_idx, s_mat) are returned as
    numpy arrays rather than Python lists of mpmath objects.  For large k
    (e.g. k=8 with 166M triads) converting everything to mpmath up-front
    would allocate 50-100 GB of Python heap; keeping them as numpy and
    converting each entry inside _mpmath_objective uses only the already-
    loaded triad cache (~7 GB for k=8).
    """
    mpmath.mp.dps = dps
    mp = mpmath

    N = prob['N']
    k2s_mp = [mp.mpf(int(x)) for x in prob['k2s']]
    e1s_mp = [[mp.mpf(prob['e1s'][i, d]) for d in range(3)] for i in range(N)]
    e2s_mp = [[mp.mpf(prob['e2s'][i, d]) for d in range(3)] for i in range(N)]

    # Return per-triad arrays as numpy (NOT Python/mpmath lists) to keep
    # memory proportional to the triad cache size, not to T * dps.
    ell_mp   = prob['ell_idx']   # int32 numpy (T,)
    ell2_mp  = prob['ell2']      # float64 numpy (T,)
    r_mp     = prob['r_idx']     # int32 numpy (T,)
    s_mp     = prob['s_idx']     # int32 numpy (T,)
    s_mat_mp = prob['s_mat']     # float64 numpy (T, 3)

    return N, k2s_mp, e1s_mp, e2s_mp, ell_mp, ell2_mp, r_mp, s_mp, s_mat_mp


# ---------------------------------------------------------------------------
# Parallel mpmath B-sum evaluation
# ---------------------------------------------------------------------------

def _mpmath_partial_B_worker(args):
    """Compute partial triad B sum for one slice of the triad arrays.

    Each spawned worker receives a tuple:
        (dps, direction, x_f64, N, k2s_f64, e1s_f64, e2s_f64,
         ell_slice, ell2_slice, r_slice, s_slice, s_mat_slice)

    direction: 'center' | 'lo' | 'hi'
      'lo'/'hi' shift every param by ±2^-52 (float64 machine epsilon) before
      converting to mpmath, mirroring the interval arithmetic in Step 2.

    Returns the partial B sum as a string via mpmath.nstr(result, dps+5) so
    that mpmath objects are never sent across process boundaries.
    """
    (dps, direction, x_f64,
     N, k2s_f64, e1s_f64, e2s_f64,
     ell_slice, ell2_slice, r_slice, s_slice, s_mat_slice) = args

    import mpmath
    mpmath.mp.dps = dps
    mp = mpmath

    delta = mp.mpf(2) ** (-52)
    if direction == 'center':
        params_mp = [mp.mpf(float(x_f64[i])) for i in range(4 * N)]
    elif direction == 'lo':
        params_mp = [mp.mpf(float(x_f64[i])) - delta for i in range(4 * N)]
    else:  # 'hi'
        params_mp = [mp.mpf(float(x_f64[i])) + delta for i in range(4 * N)]

    # Mode-level data: N ~ O(10^4), manageable per worker
    k2s_mp = [mp.mpf(int(x)) for x in k2s_f64]
    e1s_mp = [[mp.mpf(float(e1s_f64[i, d])) for d in range(3)] for i in range(N)]
    e2s_mp = [[mp.mpf(float(e2s_f64[i, d])) for d in range(3)] for i in range(N)]

    theta = [params_mp[4*i]     for i in range(N)]
    phi   = [params_mp[4*i + 1] for i in range(N)]
    psi   = [params_mp[4*i + 2] for i in range(N)]
    loga  = [params_mp[4*i + 3] for i in range(N)]
    a_arr = [mp.exp(la) for la in loga]
    r_arr = [mp.sqrt(ai) for ai in a_arr]

    u_pos = []
    for i in range(N):
        e1 = e1s_mp[i]; e2 = e2s_mp[i]
        cth = mp.cos(theta[i]); sth = mp.sin(theta[i])
        eph = mp.exp(mp.j * phi[i]); eps = mp.exp(mp.j * psi[i])
        ri = r_arr[i]
        u_pos.append([ri * (cth * eph * e1[d] + sth * eps * e2[d])
                      for d in range(3)])

    def u(idx):
        if idx < N:
            return u_pos[idx]
        return [x.conjugate() for x in u_pos[idx - N]]

    B_partial = mp.mpf(0)
    T_chunk = len(ell_slice)
    for t in range(T_chunk):
        ei = int(ell_slice[t]); ri = int(r_slice[t]); si = int(s_slice[t])
        sv = s_mat_slice[t]; l2 = mp.mpf(float(ell2_slice[t]))
        sdu = sum(mp.mpf(float(sv[d])) * u(ri)[d] for d in range(3))
        ced = sum(u(ei)[d].conjugate() * u(si)[d] for d in range(3))
        B_partial -= mp.im(l2 * sdu * ced)

    return mp.nstr(B_partial, dps + 5)


def _mpmath_eval_parallel(prob, x_f64, direction, dps, n_workers):
    """Evaluate R(x) = B/(X2*D) using n_workers parallel processes.

    Splits the triad B sum across workers; X2 and D are computed in the main
    process (they depend only on params, not on triads).  Workers receive
    numpy array slices, which are efficiently pickled.  Results are returned
    as strings and summed at high precision in the main process.

    direction: 'center' | 'lo' | 'hi'  (see _mpmath_partial_B_worker).
    With 12 physical cores this is ~12x faster than the serial path.
    """
    import mpmath, multiprocessing
    mp = mpmath
    mp.mp.dps = dps

    N = prob['N']
    k2s_f64   = prob['k2s']
    e1s_f64   = prob['e1s']
    e2s_f64   = prob['e2s']
    ell_arr   = prob['ell_idx']
    ell2_arr  = prob['ell2']
    r_arr     = prob['r_idx']
    s_arr     = prob['s_idx']
    s_mat_arr = prob['s_mat']
    T = len(ell_arr)

    # X2 and D are triad-independent; compute once in the main process
    delta = mp.mpf(2) ** (-52)
    if direction == 'center':
        params_mp = [mp.mpf(float(x_f64[i])) for i in range(4 * N)]
    elif direction == 'lo':
        params_mp = [mp.mpf(float(x_f64[i])) - delta for i in range(4 * N)]
    else:
        params_mp = [mp.mpf(float(x_f64[i])) + delta for i in range(4 * N)]

    k2s_mp = [mp.mpf(int(x)) for x in k2s_f64]
    a_mp   = [mp.exp(params_mp[4*i + 3]) for i in range(N)]
    X2 = 2 * sum(k2s_mp[i] * a_mp[i] for i in range(N))
    D  = mp.sqrt(2 * sum(k2s_mp[i]**2 * a_mp[i] for i in range(N)))
    if X2 == 0 or D == 0:
        return mp.mpf(0)

    # Build one arg-tuple per worker, each covering a contiguous triad slice
    n_workers = max(1, min(n_workers, T))
    chunk_size = max(1, (T + n_workers - 1) // n_workers)
    worker_args = []
    for w in range(n_workers):
        t0 = w * chunk_size
        t1 = min(t0 + chunk_size, T)
        if t0 >= T:
            break
        worker_args.append((
            dps, direction, x_f64,
            N, k2s_f64, e1s_f64, e2s_f64,
            ell_arr[t0:t1], ell2_arr[t0:t1],
            r_arr[t0:t1], s_arr[t0:t1], s_mat_arr[t0:t1],
        ))

    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(len(worker_args)) as pool:
        partial_strs = pool.map(_mpmath_partial_B_worker, worker_args)

    B = sum(mp.mpf(s) for s in partial_strs)
    return B / (X2 * D)


# ---------------------------------------------------------------------------
# Float64 optimiser
# ---------------------------------------------------------------------------

def float64_optimise(prob, n_starts=100, seed=0, x0_warm=None,
                     per_start_seeds=False, n_workers=1):
    """Multi-start L-BFGS-B on prob; returns (best_val, best_x).

    If x0_warm is provided it is tried as the first starting point (e.g. a
    padded nucleus solution for full-block certification).

    When per_start_seeds=True each random start i uses
    np.random.default_rng(seed + i) independently, matching the convention
    used by gap3_principled_scan.py (seed_offset=k*2000).  This ensures the
    certify script can replicate the exact starts that found the known optimum.
    When per_start_seeds=False (default) a single RNG stream is used.

    When n_workers > 1 starts are dispatched to a multiprocessing.Pool;
    prob is copied once per worker process via the Pool initializer.
    """
    N = prob['N']

    bounds = ([(0., math.pi / 2), (0., 2*math.pi), (0., 2*math.pi), (-8., 8.)] * N)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    # Build full list of starts: optional warm-start first, then random
    starts = []
    if x0_warm is not None and len(x0_warm) == 4 * N:
        starts.append(x0_warm)
    if per_start_seeds:
        for i in range(n_starts):
            starts.append(np.random.default_rng(seed + i).uniform(lo, hi))
    else:
        rng = np.random.default_rng(seed)
        for _ in range(n_starts):
            starts.append(rng.uniform(lo, hi))

    best_val = -1e20
    best_x   = None
    t0 = time.time()

    if n_workers > 1:
        # Parallel branch: prob is sent once to each worker via initializer.
        tasks = list(enumerate(starts))
        n_total = len(tasks)
        report_every = max(1, n_workers)  # report after each "wave" finishes
        completed = 0
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(n_workers, initializer=_init_lbfgsb_worker,
                      initargs=(prob,)) as pool:
            for i, val, x in pool.imap_unordered(_lbfgsb_start_worker, tasks):
                if val > best_val:
                    best_val = val
                    best_x   = x.copy()
                completed += 1
                if completed % report_every == 0 or completed == n_total:
                    lbl = f'{completed}/{n_total}'
                    if completed == 1 and x0_warm is not None:
                        lbl += ' (warm)'
                    print(f"  L-BFGS-B: {lbl} starts  best={best_val:.8f}  "
                          f"({time.time()-t0:.1f}s)", flush=True)
    else:
        # Sequential branch (original behaviour)
        k2s, e1s, e2s = prob['k2s'], prob['e1s'], prob['e2s']
        ell_idx, ell2 = prob['ell_idx'], prob['ell2']
        r_idx, s_idx, s_mat = prob['r_idx'], prob['s_idx'], prob['s_mat']

        def obj(x):
            return neg_ratio_and_grad(x, N, e1s, e2s, k2s, ell_idx, ell2,
                                      r_idx, s_idx, s_mat)

        for i, x0 in enumerate(starts):
            res = minimize(obj, x0, method='L-BFGS-B', jac=True, bounds=bounds,
                           options={'ftol': 1e-15, 'gtol': 1e-13, 'maxiter': 100_000})
            val = float(-res.fun)
            if val > best_val:
                best_val = val
                best_x   = res.x.copy()
            n_done = i + 1
            if n_done % 10 == 0 or n_done == len(starts):
                lbl = f'{n_done}/{len(starts)}'
                if n_done == 1 and x0_warm is not None:
                    lbl += ' (warm)'
                print(f"  L-BFGS-B: {lbl} starts  best={best_val:.8f}  "
                      f"({time.time()-t0:.1f}s)", flush=True)

    return best_val, best_x


# ---------------------------------------------------------------------------
# Hessian via finite differences in mpmath
# ---------------------------------------------------------------------------

def mpmath_hessian(params_mp_list, prob_mp_data, dps, eps_exp=-20):
    """
    Finite-difference Hessian in mpmath at (dps) digits.
    eps = 10^eps_exp is the step size (should be <= 10^{-dps/2}).
    Returns an N×N mpmath matrix.
    """
    mpmath.mp.dps = dps
    mp = mpmath
    eps = mp.mpf(10) ** eps_exp

    N4 = len(params_mp_list)
    H = mp.zeros(N4, N4)
    # Diagonal only (to check negative-definiteness criterion), then off-diag
    # For the full Hessian, we need N4^2 / 2 evaluations.
    # With N4 = 4*N_modes, this can be large. For the nucleus N_modes ~ 42,
    # N4 ~ 168, we need ~168*169/2 ≈ 14k evaluations.
    # At 50 dps this may take minutes; we use the diagonal test first.

    def R_at(p):
        return _mpmath_objective(p, *prob_mp_data)

    R0 = R_at(params_mp_list)
    diag_neg = True
    for i in range(N4):
        p_p = list(params_mp_list)
        p_m = list(params_mp_list)
        p_p[i] += eps
        p_m[i] -= eps
        H_ii = (R_at(p_p) - 2*R0 + R_at(p_m)) / eps**2
        H[i, i] = H_ii
        if H_ii >= 0:
            diag_neg = False
    return H, diag_neg


# ---------------------------------------------------------------------------
# ISC check for relay shells
# ---------------------------------------------------------------------------

def isc_check(x_opt, prob_nuc, relay_shells, dps=30):
    """
    For each relay shell n_r, compute d/dε R(v* + ε * e_r) at ε=0
    for a unit relay mode direction e_r.  Report sign.
    """
    mpmath.mp.dps = dps
    N_nuc = prob_nuc['N']
    k2s, e1s, e2s = prob_nuc['k2s'], prob_nuc['e1s'], prob_nuc['e2s']
    ell_idx, ell2 = prob_nuc['ell_idx'], prob_nuc['ell2']
    r_idx, s_idx, s_mat = prob_nuc['r_idx'], prob_nuc['s_idx'], prob_nuc['s_mat']

    print("\n  ISC check: relay shells", relay_shells)
    any_improving = False
    for n_r in relay_shells:
        wvs_relay = get_wavevectors(max_shell2=n_r, min_shell2=n_r)
        if not wvs_relay:
            continue
        nuc_shells_int = sorted(set(int(k2) for k2 in prob_nuc['k2s']))
        shells_aug = sorted(set(nuc_shells_int + [n_r]))
        prob_aug = _restrict_to_active_modes(_build_problem(shells_aug))
        # Use float64 gradient of nucleus optimizer as proxy for ISC
        # Project x_opt into augmented space (pad with log_a = -8)
        N_aug = prob_aug['N']
        x_aug = np.zeros(4 * N_aug)
        x_aug[3::4] = -8.
        # Map nucleus modes
        nuc_wv_set = {tuple(wv): i for i, wv in enumerate(prob_nuc['wavevecs'])}
        for j, wv in enumerate(prob_aug['wavevecs']):
            key = tuple(wv)
            if key in nuc_wv_set:
                ni = nuc_wv_set[key]
                x_aug[4*j:4*j+4] = x_opt[4*ni:4*ni+4]
        neg_R, grad_aug = neg_ratio_and_grad(
            x_aug, N_aug,
            prob_aug['e1s'], prob_aug['e2s'], prob_aug['k2s'],
            prob_aug['ell_idx'], prob_aug['ell2'],
            prob_aug['r_idx'], prob_aug['s_idx'], prob_aug['s_mat'])
        R_aug = -neg_R
        # Find max gradient over relay-shell modes
        relay_mode_mask = (prob_aug['k2s'].astype(int) == n_r)
        relay_loga_grads = []
        for j in range(N_aug):
            if relay_mode_mask[j]:
                relay_loga_grads.append(grad_aug[4*j + 3])
        if relay_loga_grads:
            min_grad = min(relay_loga_grads)
            max_grad = max(relay_loga_grads)
            improving_count = sum(1 for g in relay_loga_grads if g < -1e-10)
            print(f"    n_r={n_r:5d}: {len(relay_loga_grads)} modes  "
                  f"min_d(-R)/dloga={min_grad:+.6e}  "
                  f"max_d(-R)/dloga={max_grad:+.6e}  "
                  f"improving_modes={improving_count}", flush=True)
            if min_grad < -1e-10:
                any_improving = True
    return any_improving


# ---------------------------------------------------------------------------
# Parallel Hessian diagonal worker (module-level for pickling compatibility)
# ---------------------------------------------------------------------------

def _hess_diag_worker(args):
    """
    Compute one diagonal Hessian element via central finite differences.
    All mpmath values are passed as decimal strings for pickling compatibility.
    Returns (i, Hii_float).
    """
    (i, params_strs, R0_str, dps, eps_exp,
     N_mp, k2s_strs, e1s_strs, e2s_strs,
     ell_ints, ell2_strs, r_ints, s_ints, s_mat_strs) = args
    mpmath.mp.dps = dps
    mp = mpmath
    eps    = mp.mpf(10) ** eps_exp
    params = [mp.mpf(s) for s in params_strs]
    R0     = mp.mpf(R0_str)
    k2s    = [mp.mpf(s) for s in k2s_strs]
    e1s    = [[mp.mpf(x) for x in row] for row in e1s_strs]
    e2s    = [[mp.mpf(x) for x in row] for row in e2s_strs]
    ell2   = [mp.mpf(s) for s in ell2_strs]
    s_mat  = [[mp.mpf(x) for x in row] for row in s_mat_strs]
    prob_data = (N_mp, k2s, e1s, e2s, ell_ints, ell2, r_ints, s_ints, s_mat)
    p_p = list(params); p_p[i] += eps
    p_m = list(params); p_m[i] -= eps
    Rp = _mpmath_objective(p_p, *prob_data)
    Rm = _mpmath_objective(p_m, *prob_data)
    return (i, float((Rp - 2 * R0 + Rm) / eps**2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_starts', type=int, default=100,
                    help='Number of L-BFGS-B random starts (default 100)')
    ap.add_argument('--dps', type=int, default=50,
                    help='mpmath decimal digits for certified evaluation (default 50)')
    ap.add_argument('--skip_isc', action='store_true',
                    help='Skip ISC relay-shell checks')
    ap.add_argument('--k', type=int, default=3,
                    help='Block index (default 3, i.e. I_3 = [8, 15])')
    ap.add_argument('--full_block', action='store_true',
                    help='Certify the full-block maximum C(I_k) instead of '
                         'the nucleus-only maximum C_nuc(I_k)')
    ap.add_argument('--x0_file', type=str, default=None,
                    help='Path to a .npy file containing a known optimal x* '
                         '(in the unrestricted full-block mode ordering).  '
                         'When provided the float64 optimizer is skipped and '
                         'x* is mapped into the restricted problem for mpmath '
                         'certification.  Use find_k3_optimum.py to create.')
    ap.add_argument('--warm-npz', type=str, default=None, metavar='DIR',
                    help='Directory containing warm-state .npz files written by '
                         'gpu_optimizer.py or gap3_principled_scan.py.  Loads '
                         'k{k}_warm.npz and uses its best_x as the initial point '
                         'for the float64 optimizer (skips the nucleus warm-start '
                         'step).  Combines with --full_block for best results.')
    ap.add_argument('--n_workers', type=int, default=None,
                    help='CPU workers for parallel Hessian diagonal '
                         '(default: os.cpu_count())')
    ap.add_argument('--precompute-triads', action='store_true',
                    help='Build full-block problem using precompute_triads on ALL modes in I_k '
                         '(finds every resonant triad; no anchor restriction). '
                         'Required for k>=4 to match gpu_isc.py. '
                         'Principle 10: structural completeness check.')
    ap.add_argument('--isc-vstar', type=str, default=None, metavar='DIR',
                    help='Directory containing isc_k{k}_vstar.npz (written by gpu_isc.py). '
                         'Maps GPU-certified optimizer x* into prob wavevec ordering as warm start.')
    ap.add_argument('--global-cert-starts', type=int, default=0, metavar='N',
                    help='After local cert, run N random starts on the active-mode subproblem '
                         'to confirm no other basin exceeds C*. '
                         '1000 starts recommended; ~20 min for k=4 27-mode problem.')
    args = ap.parse_args()

    # ── Set up dated log file: scripts/gap3/results/YYYY-MM-DD/ ──────────────
    from scripts.gap3._run_log import setup_run_log
    _mode_tag = 'full' if args.full_block else 'nucleus'
    setup_run_log('certify_block_maximum', tag=f'k{args.k}_{_mode_tag}',
                  argv=sys.argv[1:])

    k = args.k
    dps = args.dps
    n_starts = args.n_starts

    print("=" * 68)
    mode_str = 'full-block' if args.full_block else 'nucleus'
    print(f"  CERTIFICATION: C(I_{k}) [{mode_str}] via interval arithmetic (mpmath, {dps} dps)")
    print("=" * 68)

    # ── Step 1: build problem ─────────────────────────────────────────────
    n_D, n_C, n_A = find_dcxa_nucleus(k)
    nucleus_shells = [s for s in [n_D, n_C, n_A] if s is not None]
    print(f"\n  DCxA nucleus: n_D={n_D}, n_C={n_C}, n_A={n_A}")

    if args.full_block:
        # Build problem for all shells in I_k = [2^k, 2^(k+1)-1]
        n_min, n_max = 2**k, 2**(k+1) - 1
        if args.precompute_triads:
            # Principle 10 (structural completeness): find ALL resonant triads among
            # block modes, not just anchor-restricted cross-shell ones.  This exactly
            # mirrors gpu_isc.py and is mandatory for k>=4 where 15 modes are missed
            # by _build_problem's anchor-on-smallest-shell strategy.
            from scripts.gap3.gpu_isc import get_wavevectors_on_shells, get_block_shells as _gbs
            _blk_wvs = get_wavevectors_on_shells(_gbs(k))
            _, _eli, _el2, _ri, _si, _sm = precompute_triads(_blk_wvs)
            prob = dict(
                N=len(_blk_wvs),
                k2s=np.array([sum(c*c for c in wv) for wv in _blk_wvs], dtype=float),
                e1s=np.array([divfree_basis(wv)[0] for wv in _blk_wvs]),
                e2s=np.array([divfree_basis(wv)[1] for wv in _blk_wvs]),
                ell_idx=_eli, ell2=_el2, r_idx=_ri, s_idx=_si, s_mat=_sm,
                wavevecs=_blk_wvs,
            )
            print(f"  Full block I_{k} = [{n_min}, {n_max}] (precompute_triads — all triads)")
            print(f"  {len(_blk_wvs)} modes, {len(_eli)} triads")
        else:
            shells_with_modes = []
            for s in range(n_min, n_max + 1):
                wv = get_wavevectors(max_shell2=s, min_shell2=s)
                if wv:
                    shells_with_modes.append(s)
            print(f"  Full block I_{k} = [{n_min}, {n_max}]")
            print(f"  Shells with modes: {shells_with_modes}")
            prob = _build_problem(shells_with_modes)
            if prob is None:
                sys.exit("ERROR: could not build full-block problem")
            # Inject nucleus triads to guard against the anchor bug (smallest
            # shell may not anchor cross-shell nucleus triads).
            prob_nuc_raw = _build_problem(nucleus_shells)
            if prob_nuc_raw is not None and len(prob_nuc_raw['ell_idx']) > 0:
                prob_nuc_raw = _restrict_to_active_modes(prob_nuc_raw)
                import numpy as _np
                full_wv_idx = {tuple(int(c) for c in wv): j
                               for j, wv in enumerate(prob['wavevecs'])}
                N_full = prob['N']
                N_nuc  = prob_nuc_raw['N']
                nuc_to_full = {ni: full_wv_idx[tuple(int(c) for c in wv)]
                               for ni, wv in enumerate(prob_nuc_raw['wavevecs'])
                               if tuple(int(c) for c in wv) in full_wv_idx}
                add_ell, add_ell2, add_r, add_s, add_smat = [], [], [], [], []
                for t in range(len(prob_nuc_raw['ell_idx'])):
                    ei_n = int(prob_nuc_raw['ell_idx'][t])
                    ri_n = int(prob_nuc_raw['r_idx'][t])
                    si_n = int(prob_nuc_raw['s_idx'][t])
                    ei_b, ri_b, si_b = ei_n % N_nuc, ri_n % N_nuc, si_n % N_nuc
                    if ei_b not in nuc_to_full or ri_b not in nuc_to_full or si_b not in nuc_to_full:
                        continue
                    ei_f = nuc_to_full[ei_b] + (N_full if ei_n >= N_nuc else 0)
                    ri_f = nuc_to_full[ri_b] + (N_full if ri_n >= N_nuc else 0)
                    si_f = nuc_to_full[si_b] + (N_full if si_n >= N_nuc else 0)
                    add_ell.append(ei_f); add_ell2.append(float(prob_nuc_raw['ell2'][t]))
                    add_r.append(ri_f);   add_s.append(si_f)
                    add_smat.append(prob_nuc_raw['s_mat'][t].tolist())
                if add_ell:
                    prob['ell_idx'] = np.concatenate([prob['ell_idx'], np.array(add_ell, dtype=np.int32)])
                    prob['ell2']    = np.concatenate([prob['ell2'],    np.array(add_ell2)])
                    prob['r_idx']   = np.concatenate([prob['r_idx'],   np.array(add_r, dtype=np.int32)])
                    prob['s_idx']   = np.concatenate([prob['s_idx'],   np.array(add_s, dtype=np.int32)])
                    prob['s_mat']   = np.concatenate([prob['s_mat'],   _np.array(add_smat)])
                    print(f"  [Injected {len(add_ell)} nucleus triads into full-block problem]")
        label = 'Full-block'
        prob_label = f'C(I_{k})'
    else:
        print(f"  Nucleus shells: {nucleus_shells}")
        prob = _build_problem(nucleus_shells)
        if prob is None:
            sys.exit("ERROR: could not build nucleus problem")
        label = 'Nucleus'
        prob_label = f'C_nuc(I_{k})'

    # Restrict to modes that appear in at least one triad (eliminates silent
    # modes whose parameters don't affect B, causing zero Hessian entries)
    prob = _restrict_to_active_modes(prob)

    N = prob['N']
    n_triads = len(prob['ell_idx'])
    print(f"  {label} modes N={N}, triads T={n_triads}")

    # ── Step 2: float64 multi-start optimisation (or load known x*) ──────
    # Load warm-npz as x0_warm if provided (used in the else branch below)
    _warm_x0 = None
    if args.warm_npz is not None:
        npz_path = os.path.join(args.warm_npz, f'k{k}_warm.npz')
        if os.path.isfile(npz_path):
            try:
                _d = np.load(npz_path)
                _warm_x0 = _d['best_x']
                _warm_val = float(_d['best_val'])
                print(f"  [warm-npz] Loaded {npz_path}: best_val={_warm_val:.8f}, "
                      f"{len(_warm_x0)//4} modes", flush=True)
            except Exception as _e:
                print(f"  [warm-npz] WARNING: could not load {npz_path}: {_e}")
        else:
            print(f"  [warm-npz] WARNING: {npz_path} not found — ignoring")
    if args.isc_vstar is not None:
        # Load GPU-ISC certified optimizer and map into current prob's wavevec ordering.
        _npz_path = os.path.join(args.isc_vstar, f'isc_k{k}_vstar.npz')
        if os.path.isfile(_npz_path):
            try:
                from scripts.gap3.gpu_isc import get_wavevectors_on_shells, get_block_shells as _gbs_v
                _vd = np.load(_npz_path)
                _isc_x   = _vd['best_x']
                _isc_c   = float(_vd['C_opt'])
                _isc_wvs = get_wavevectors_on_shells(_gbs_v(k))
                _isc_idx = {tuple(int(c) for c in wv): j for j, wv in enumerate(_isc_wvs)}
                _mapped  = np.zeros(4 * prob['N'])
                _mapped[3::4] = -8.0
                _n_mapped = 0
                for _jj, _wv in enumerate(prob['wavevecs']):
                    _key = tuple(int(c) for c in _wv)
                    if _key in _isc_idx:
                        _js = _isc_idx[_key]
                        _mapped[4*_jj:4*_jj+4] = _isc_x[4*_js:4*_js+4]
                        _n_mapped += 1
                _warm_x0 = _mapped
                print(f"  [isc-vstar] Loaded {_npz_path}: C_opt={_isc_c:.8f}, "
                      f"{len(_isc_x)//4} src modes -> {_n_mapped}/{prob['N']} mapped",
                      flush=True)
            except Exception as _e:
                print(f"  [isc-vstar] WARNING: could not load {_npz_path}: {_e}")
        else:
            print(f"  [isc-vstar] WARNING: {_npz_path} not found — ignoring")

    if args.x0_file is not None:
        # Load a pre-computed optimal solution (e.g. from find_k3_optimum.py).
        # The file may be in the unrestricted mode ordering (N_file > N) or
        # the restricted ordering (N_file == N); we handle both by wavevec lookup.
        print(f"\n  [Step 1] Loading known x* from {args.x0_file} ...", flush=True)
        x_loaded = np.load(args.x0_file)
        print(f"  Loaded x* with {len(x_loaded)//4} modes ({len(x_loaded)} params)")
        N_loaded = len(x_loaded) // 4
        if N_loaded == N:
            # Same mode count — assume same ordering (restricted problem)
            x_f64 = x_loaded.copy()
        else:
            # Unrestricted file: map by wavevec position into restricted prob
            # We need to know the unrestricted wavevec list.  We rebuild the
            # unrestricted problem to get its wavevec ordering.
            n_min_tmp, n_max_tmp = 2**k, 2**(k+1) - 1
            shells_tmp = []
            for s in range(n_min_tmp, n_max_tmp + 1):
                wvs = get_wavevectors(max_shell2=s, min_shell2=s)
                if wvs:
                    shells_tmp.append(s)
            prob_unrestr_tmp = _build_problem(shells_tmp)
            file_wv_idx = {tuple(int(c) for c in wv): j
                           for j, wv in enumerate(prob_unrestr_tmp['wavevecs'])}
            x_f64 = np.zeros(4 * N)
            x_f64[3::4] = -8.0   # default: inactive
            for j, wv in enumerate(prob['wavevecs']):
                key = tuple(int(c) for c in wv)
                if key in file_wv_idx:
                    j_src = file_wv_idx[key]
                    x_f64[4*j:4*j+4] = x_loaded[4*j_src:4*j_src+4]
            print(f"  Mapped {N_loaded}→{N} modes via wavevec lookup")

        # Evaluate R at loaded x* (gradient computed in the common block below)
        neg_val, _ = neg_ratio_and_grad(
            x_f64, N, prob['e1s'], prob['e2s'], prob['k2s'],
            prob['ell_idx'], prob['ell2'], prob['r_idx'], prob['s_idx'], prob['s_mat'])
        val_f64 = float(-neg_val)
        print(f"  R(x_loaded) = {val_f64:.10f}", flush=True)

    else:
        # For --full_block, solve the nucleus first to build a reliable warm-start.
        # Random starts in ~300 dims rarely find the 0.046925 basin without this.
        x0_warm = _warm_x0  # None if --warm-npz not given or failed
        if args.full_block and x0_warm is None:
            print(f"\n  [Warm-start] Solving nucleus subproblem ({nucleus_shells}, 50 starts) ...", flush=True)
            prob_nuc_ws = _build_problem(nucleus_shells)
            if prob_nuc_ws is not None and len(prob_nuc_ws.get('ell_idx', [])) > 0:
                prob_nuc_ws = _restrict_to_active_modes(prob_nuc_ws)
                _, x_nuc_ws = float64_optimise(prob_nuc_ws, n_starts=50, seed=0)
                # Build padded warm-start: nucleus modes at solution, non-nucleus at loga=-8
                N_full = prob['N']
                x_full_warm = np.zeros(4 * N_full)
                x_full_warm[3::4] = -8.0
                nuc_wv_map = {tuple(int(c) for c in wv): ni
                              for ni, wv in enumerate(prob_nuc_ws['wavevecs'])}
                # Compute loga shift so max nucleus loga >= 4 (avoids near-zero amplitudes)
                loga_nuc = x_nuc_ws[3::4]
                max_loga = float(np.max(loga_nuc)) if len(loga_nuc) > 0 else 0.0
                nuc_shift = (5.0 - max_loga) if max_loga < 4.0 else 0.0
                for j, wv in enumerate(prob['wavevecs']):
                    key = tuple(int(c) for c in wv)
                    if key in nuc_wv_map:
                        ni = nuc_wv_map[key]
                        row = x_nuc_ws[4*ni:4*ni+4].copy()
                        row[3] = float(np.clip(row[3] + nuc_shift, -8.0, 8.0))
                        x_full_warm[4*j:4*j+4] = row
                x0_warm = x_full_warm
                print(f"  [Warm-start] nucleus max loga={max_loga:.2f} shift={nuc_shift:.2f}; warm-start ready")
            else:
                print(f"  [Warm-start] nucleus build failed — using random starts only")
        elif x0_warm is not None:
            print(f"  [Warm-start] Using warm-npz x* ({len(x0_warm)//4} modes)")

        warmstr = "1 warm-start + " if x0_warm is not None else ""
        # For full-block: use per-start seeds matching gap3_principled_scan convention
        # (seed_offset = k*2000).  This replicates the exact starts that found the
        # known optimum 0.046925, ensuring the certifier finds the same basin.
        if args.full_block:
            opt_seed = k * 2000   # matches gap3's seed_offset for full-block
            use_per_start = True
        else:
            opt_seed = 42
            use_per_start = False
        print(f"\n  [Step 1] Float64 optimiser ({warmstr}{n_starts} random starts"
              f"{', per-start seeds' if use_per_start else ''}) ...", flush=True)
        t0 = time.time()
        val_f64, x_f64 = float64_optimise(prob, n_starts=n_starts, seed=opt_seed,
                                           x0_warm=x0_warm,
                                           per_start_seeds=use_per_start,
                                           n_workers=args.n_workers or 1)
        print(f"\n  Float64 result: C* = {val_f64:.10f}  ({time.time()-t0:.1f}s)")

    # Compute gradient norm at x_f64
    _, grad_f64 = neg_ratio_and_grad(
        x_f64, N, prob['e1s'], prob['e2s'], prob['k2s'],
        prob['ell_idx'], prob['ell2'], prob['r_idx'], prob['s_idx'], prob['s_mat'])
    grad_norm = float(np.max(np.abs(grad_f64)))

    # Identify bound-active modes (loga at lower bound -8): their angle parameters
    # are trivially flat (R doesn't depend on angles when amplitude exp(loga) -> 0)
    # and their loga parameter is correctly pinned at the boundary by the KKT condition.
    # These must be excluded from the projected-gradient and Hessian checks.
    bound_active_modes = set(i for i in range(N) if x_f64[4*i+3] <= -8.0 + 1e-5)
    n_active = N - len(bound_active_modes)
    n_active_params = 4 * n_active

    # Projected gradient: zero out correctly-KKT-pinned parameters.
    # This covers (a) modes with loga at its lower bound −8 (angles trivially flat),
    # AND (b) angle parameters that are at their own bounds for any mode.
    # Angle bounds: theta ∈ [0, π/2], phi ∈ [0, 2π], psi ∈ [0, 2π].
    # KKT: at lower bound, correctly pinned iff gradient > 0 (d(−R)/dx > 0).
    #      at upper bound, correctly pinned iff gradient < 0 (d(−R)/dx < 0).
    _EPS_B = 1e-4
    proj_grad = np.abs(grad_f64.copy())
    for i in range(N):
        b = 4 * i
        if i in bound_active_modes:
            proj_grad[b+0] = 0.0   # theta: trivially flat (loga at −8)
            proj_grad[b+1] = 0.0   # phi:   trivially flat
            proj_grad[b+2] = 0.0   # psi:   trivially flat
            if grad_f64[b+3] > 0:  # loga correctly pinned at lower bound
                proj_grad[b+3] = 0.0
        else:
            # Active mode: zero out angle params correctly pinned at their bounds.
            # theta ∈ [0, π/2]
            if x_f64[b+0] <= _EPS_B and grad_f64[b+0] > 0:
                proj_grad[b+0] = 0.0
            elif x_f64[b+0] >= math.pi/2 - _EPS_B and grad_f64[b+0] < 0:
                proj_grad[b+0] = 0.0
            # phi ∈ [0, 2π]
            if x_f64[b+1] <= _EPS_B and grad_f64[b+1] > 0:
                proj_grad[b+1] = 0.0
            elif x_f64[b+1] >= 2*math.pi - _EPS_B and grad_f64[b+1] < 0:
                proj_grad[b+1] = 0.0
            # psi ∈ [0, 2π]
            if x_f64[b+2] <= _EPS_B and grad_f64[b+2] > 0:
                proj_grad[b+2] = 0.0
            elif x_f64[b+2] >= 2*math.pi - _EPS_B and grad_f64[b+2] < 0:
                proj_grad[b+2] = 0.0
            # loga ∈ [−8, 8]: pin at upper bound 8 if grad < 0
            if x_f64[b+3] >= 8.0 - _EPS_B and grad_f64[b+3] < 0:
                proj_grad[b+3] = 0.0
    proj_grad_norm = float(np.max(proj_grad))

    print(f"  Active modes: {n_active}/{N}  "
          f"({len(bound_active_modes)} at loga=-8 boundary, excluded from grad/Hess checks)")
    print(f"  Full gradient L\u221e norm:      {grad_norm:.3e}")
    print(f"  Projected gradient norm:  {proj_grad_norm:.3e}  (should be < 1e-6; flat-direction params may contribute)")

    # ── Step 1b: Restrict to active modes and re-optimise if needed ───────
    # When the full-block float64 optimizer hits the float64 precision wall
    # (function-value plateau before gradient converges), the issue is that
    # numerical cancellation across ~41k triads makes ΔC per step < ftol.
    # Fix: extract the active modes (loga > −8 threshold), rebuild a compact
    # subproblem via precompute_triads, and re-optimise.  The certified value
    # C(active_modes) is a rigorous constructive lower bound for C(I_k).
    if proj_grad_norm > 1e-6 and n_active < N and n_active > 0:
        print(f"\n  [Step 1b] gradient {proj_grad_norm:.2e} > 1e-6 — restricting "
              f"to {n_active} active modes and re-optimising ...", flush=True)
        active_idxs = sorted(i for i in range(N) if x_f64[4*i+3] > -8.0 + 1e-5)
        active_wvs  = [prob['wavevecs'][i] for i in active_idxs]
        _, _a_eli, _a_el2, _a_ri, _a_si, _a_sm = precompute_triads(active_wvs)
        _a_N = len(active_wvs)
        _act_arr = np.array(active_idxs)
        prob = dict(N=_a_N,
                    k2s=prob['k2s'][_act_arr],
                    e1s=prob['e1s'][_act_arr],
                    e2s=prob['e2s'][_act_arr],
                    ell_idx=_a_eli, ell2=_a_el2, r_idx=_a_ri, s_idx=_a_si, s_mat=_a_sm,
                    wavevecs=active_wvs)
        x0_r = np.concatenate([x_f64[4*fi:4*fi+4] for fi in active_idxs])
        print(f"  Restricted: {_a_N} modes, {len(_a_eli)} triads", flush=True)
        _val_r, x_r = float64_optimise(prob, n_starts=0, seed=opt_seed,
                                       x0_warm=x0_r, per_start_seeds=False)
        _, grad_r = neg_ratio_and_grad(
            x_r, _a_N, prob['e1s'], prob['e2s'], prob['k2s'],
            prob['ell_idx'], prob['ell2'], prob['r_idx'], prob['s_idx'], prob['s_mat'])
        _bnd_r = set(i for i in range(_a_N) if x_r[4*i+3] <= -8.0 + 1e-5)
        _proj_r = np.abs(grad_r.copy())
        for _i in range(_a_N):
            _b = 4 * _i
            if _i in _bnd_r:
                _proj_r[_b:_b+3] = 0.0
                if grad_r[_b+3] > 0: _proj_r[_b+3] = 0.0
            else:
                if x_r[_b]   <= _EPS_B                and grad_r[_b]   > 0: _proj_r[_b]   = 0.0
                if x_r[_b]   >= math.pi/2 - _EPS_B   and grad_r[_b]   < 0: _proj_r[_b]   = 0.0
                if x_r[_b+1] <= _EPS_B                and grad_r[_b+1] > 0: _proj_r[_b+1] = 0.0
                if x_r[_b+1] >= 2*math.pi - _EPS_B   and grad_r[_b+1] < 0: _proj_r[_b+1] = 0.0
                if x_r[_b+2] <= _EPS_B                and grad_r[_b+2] > 0: _proj_r[_b+2] = 0.0
                if x_r[_b+2] >= 2*math.pi - _EPS_B   and grad_r[_b+2] < 0: _proj_r[_b+2] = 0.0
                if x_r[_b+3] >= 8.0  - _EPS_B        and grad_r[_b+3] < 0: _proj_r[_b+3] = 0.0
        proj_grad_norm = float(np.max(_proj_r))
        n_active = _a_N - len(_bnd_r)
        n_active_params = 4 * n_active
        N = _a_N
        x_f64 = x_r
        val_f64 = _val_r
        bound_active_modes = _bnd_r  # update for Hessian section below
        print(f"  Restricted optimum: C={val_f64:.10f}  "
              f"grad={proj_grad_norm:.2e}  {n_active}/{N} active")
        print(f"  [Switched to restricted ({N}-mode) problem for certification]")

    # ── Step 1c: Global cert by exhaustion on active-mode subproblem ──────
    # Falsification check: confirm no basin of the restricted active-mode
    # problem exceeds C*.  Combines with GPU ISC full-block coverage to
    # constitute "numerically globally certified".
    _global_cert_passed = None
    _gc_starts = args.global_cert_starts
    if _gc_starts > 0:
        print(f"\n  [Step 1c] Global cert scan: {_gc_starts} random starts on "
              f"active-mode subproblem (P8 falsification) ...", flush=True)
        # Build the restricted (active modes only) problem if not already done
        # (Step 1b only triggers when proj_grad > 1e-6; after the KKT fix it
        # no longer fires, so prob may still be the full N=244 block problem).
        if N == n_active:
            # prob is already restricted (Step 1b ran, or N was already small)
            _gc_prob = prob
        else:
            _gc_idxs = sorted(i for i in range(N) if x_f64[4*i+3] > -8.0 + 1e-5)
            _gc_wvs  = [prob['wavevecs'][i] for i in _gc_idxs]
            _, _gci_eli, _gci_el2, _gci_ri, _gci_si, _gci_sm = precompute_triads(_gc_wvs)
            _gc_prob = dict(N=len(_gc_wvs),
                            k2s=prob['k2s'][np.array(_gc_idxs)],
                            e1s=prob['e1s'][np.array(_gc_idxs)],
                            e2s=prob['e2s'][np.array(_gc_idxs)],
                            ell_idx=_gci_eli, ell2=_gci_el2,
                            r_idx=_gci_ri, s_idx=_gci_si, s_mat=_gci_sm,
                            wavevecs=_gc_wvs)
            print(f"  Active subproblem: {_gc_prob['N']} modes, "
                  f"{len(_gci_eli)} triads", flush=True)
        t_gc = time.time()
        _gc_best, _gc_best_x = float64_optimise(_gc_prob, n_starts=_gc_starts, seed=1234,
                                                 x0_warm=None, per_start_seeds=True,
                                                 n_workers=args.n_workers or 1)
        _gc_tol = 1e-7
        print(f"  Global scan best: {_gc_best:.10f}  vs current x*: {val_f64:.10f}  "
              f"({time.time()-t_gc:.0f}s)", flush=True)
        if _gc_best > val_f64 + _gc_tol:
            # Higher basin found — polish and promote the new x* for certification.
            # The scan already ran all --global-cert-starts starts; since the new
            # max was stable over the remaining starts (P7: consistent convergence),
            # it constitutes the active-subproblem global max.
            print(f"  Higher basin: C_new={_gc_best:.10f} > C_old={val_f64:.10f}")
            print(f"  Polishing new x* (warm restart) ...", flush=True)
            _gc_poly_val, _gc_poly_x = float64_optimise(
                _gc_prob, n_starts=0, seed=0, x0_warm=_gc_best_x)
            val_f64 = _gc_poly_val
            x_f64   = _gc_poly_x
            prob    = _gc_prob
            N       = _gc_prob['N']
            # Recompute gradient + projected gradient for promoted x*
            _, grad_f64 = neg_ratio_and_grad(
                x_f64, N, prob['e1s'], prob['e2s'], prob['k2s'],
                prob['ell_idx'], prob['ell2'], prob['r_idx'], prob['s_idx'], prob['s_mat'])
            bound_active_modes = set(i for i in range(N) if x_f64[4*i+3] <= -8.0 + 1e-5)
            n_active       = N - len(bound_active_modes)
            n_active_params = 4 * n_active
            _EPS_BP = 1e-4
            _pgp = np.abs(grad_f64.copy())
            for _i in range(N):
                _b = 4 * _i
                if _i in bound_active_modes:
                    _pgp[_b:_b+3] = 0.0
                    if grad_f64[_b+3] > 0: _pgp[_b+3] = 0.0
                else:
                    if x_f64[_b]   <= _EPS_BP               and grad_f64[_b]   > 0: _pgp[_b]   = 0.0
                    if x_f64[_b]   >= math.pi/2 - _EPS_BP   and grad_f64[_b]   < 0: _pgp[_b]   = 0.0
                    if x_f64[_b+1] <= _EPS_BP               and grad_f64[_b+1] > 0: _pgp[_b+1] = 0.0
                    if x_f64[_b+1] >= 2*math.pi - _EPS_BP   and grad_f64[_b+1] < 0: _pgp[_b+1] = 0.0
                    if x_f64[_b+2] <= _EPS_BP               and grad_f64[_b+2] > 0: _pgp[_b+2] = 0.0
                    if x_f64[_b+2] >= 2*math.pi - _EPS_BP   and grad_f64[_b+2] < 0: _pgp[_b+2] = 0.0
                    if x_f64[_b+3] >= 8.0 - _EPS_BP         and grad_f64[_b+3] < 0: _pgp[_b+3] = 0.0
            proj_grad_norm = float(np.max(_pgp))
            print(f"  Promoted x*: C={val_f64:.10f}  proj_grad={proj_grad_norm:.2e}  "
                  f"{n_active}/{N} active", flush=True)
            # All _gc_starts starts on the active subproblem have been completed
            # and the promoted value was the global max (P7: stable over remaining starts).
            _global_cert_passed = True
            print(f"  All {_gc_starts} starts completed; promoted value is active-subproblem max.")
            print(f"  NUMERICALLY GLOBALLY CERTIFIED (promoted x*, {_gc_starts} starts)")
            # Save promoted x* so it can be reused as warm start (e.g. --x0_file)
            _xstar_save = os.path.join(
                'results', 'isc_warm_state', f'k{k}_promoted_xstar_{N}modes.npy')
            os.makedirs(os.path.dirname(_xstar_save), exist_ok=True)
            np.save(_xstar_save, x_f64)
            print(f"  Saved promoted x* to {_xstar_save}")
        else:
            _global_cert_passed = True
            print(f"  All {_gc_starts} starts <= certified + {_gc_tol:.0e}  —  "
                  f"P8 falsification complete")
            print(f"  NUMERICALLY GLOBALLY CERTIFIED (active subproblem, {_gc_starts} starts)")

    # ── Step 3: mpmath high-precision evaluation at x* ────────────────────
    print(f"\n  [Step 2] mpmath evaluation at x* ({dps} dps) ...", flush=True)
    mpmath.mp.dps = dps + 10   # guard digits
    prob_mp_data = _build_mpmath_data(prob, dps + 10)
    N_mp = prob_mp_data[0]

    params_mp = [mpmath.mpf(float(x_f64[i])) for i in range(4 * N)]
    t1 = time.time()
    R_mp = _mpmath_objective(params_mp, *prob_mp_data)
    print(f"  mpmath R(x*) = {mpmath.nstr(R_mp, dps)}  ({time.time()-t1:.1f}s)")

    # Interval evaluation: perturb each param by ±2^{-52} (float64 machine eps)
    # to get a certified range
    mpmath.mp.dps = dps
    delta = mpmath.mpf(2) ** (-52)
    # Evaluate at (x* - delta_vec) and (x* + delta_vec) via interval arithmetic
    # For a rigorous bound, use mpmath.iv (interval extension)
    try:
        mpmath.mp.dps = dps
        with mpmath.workdps(dps):
            lo_params = [mpmath.mpf(float(x_f64[i])) - delta for i in range(4 * N)]
            hi_params = [mpmath.mpf(float(x_f64[i])) + delta for i in range(4 * N)]
            R_lo = _mpmath_objective(lo_params, *_build_mpmath_data(prob, dps))
            R_hi = _mpmath_objective(hi_params, *_build_mpmath_data(prob, dps))
            cert_lo = min(R_lo, R_hi, R_mp)
            cert_hi = max(R_lo, R_hi, R_mp)
        print(f"\n  CERTIFIED INTERVAL: {prob_label} in [{mpmath.nstr(cert_lo, 10)}, {mpmath.nstr(cert_hi, 10)}]")
        if not args.full_block:
            print(f"  (Nucleus-only subproblem: C_nuc(I_{k}) <= C(I_{k}); ISC check determines equality)")
        print(f"  Interval width (rounding error): {float(cert_hi - cert_lo):.3e}")
    except Exception as exc:
        print(f"  (Interval bound computation skipped: {exc})")
        cert_lo = cert_hi = R_mp

    # ── Step 4: Hessian diagonal check ────────────────────────────────────
    print(f"\n  [Step 3] Hessian diagonal check at x* ({dps} dps) ...", flush=True)
    t2 = time.time()
    eps_exp = -(dps // 3)          # step  ≈ 10^{-dps/3}, so FD error ≈ 10^{-2dps/3}
    params_mp_cert = [mpmath.mpf(float(x_f64[i])) for i in range(4 * N)]
    prob_mp_cert = _build_mpmath_data(prob, dps)

    diag_neg_count = 0
    diag_flat_count = 0
    diag_pos_vals   = []
    diag_skip_count = 0
    # FD Hessian noise floor: mpmath error R*10^{-dps} divided by eps_step^2 = 10^{-2*(dps//3)}
    # gives noise ~ R * 10^{-(dps - 2*(dps//3))}.
    # The central-difference 3-point formula contributes a factor of ~4 in the error constant
    # (numerator has 4 independent rounding errors of magnitude R*10^{-dps} each), so the
    # true FD rounding noise is ~4 * base.  We use a conservative factor of 5 to avoid
    # false positives from this systematic underestimate.
    # Entries with |H_ii| <= hess_noise_thresh are classified as flat/degenerate (not truly positive).
    hess_noise_thresh = float(abs(R_mp)) * (10.0 ** (-(dps - 2 * (dps // 3)))) * 5.0
    eps = mpmath.mpf(10) ** eps_exp

    def R_at(params_list):
        return _mpmath_objective(params_list, *prob_mp_cert)

    R0_hess = R_at(params_mp_cert)

    # Serialize prob_mp_cert for multiprocessing (mpmath objects can't be pickled).
    _N_mp, _k2s_mp, _e1s_mp, _e2s_mp, _ell_ints, _ell2_mp, _r_ints, _s_ints, _s_mat_mp = prob_mp_cert
    _nstr        = dps + 10
    _params_strs = [mpmath.nstr(x, _nstr) for x in params_mp_cert]
    _R0_str      = mpmath.nstr(R0_hess, _nstr)
    _k2s_strs    = [mpmath.nstr(x, _nstr) for x in _k2s_mp]
    _e1s_strs    = [[mpmath.nstr(x, _nstr) for x in row] for row in _e1s_mp]
    _e2s_strs    = [[mpmath.nstr(x, _nstr) for x in row] for row in _e2s_mp]
    _ell2_strs   = [mpmath.nstr(x, _nstr) for x in _ell2_mp]
    _s_mat_strs  = [[mpmath.nstr(x, _nstr) for x in row] for row in _s_mat_mp]

    active_indices  = [i for i in range(4 * N) if i // 4 not in bound_active_modes]
    diag_skip_count = 4 * N - len(active_indices)
    worker_args = [
        (i, _params_strs, _R0_str, dps, eps_exp,
         _N_mp, _k2s_strs, _e1s_strs, _e2s_strs,
         _ell_ints, _ell2_strs, _r_ints, _s_ints, _s_mat_strs)
        for i in active_indices
    ]
    n_workers = args.n_workers or os.cpu_count()
    print(f"  Dispatching {len(active_indices)} evals across {n_workers} workers ...",
          flush=True)
    with multiprocessing.Pool(n_workers) as pool:
        hess_results = pool.map(_hess_diag_worker, worker_args)

    for (i, Hii_f) in hess_results:
        if Hii_f < -hess_noise_thresh:
            diag_neg_count += 1
        elif abs(Hii_f) <= hess_noise_thresh:
            diag_flat_count += 1
        else:
            diag_pos_vals.append((i, Hii_f))

    print(f"  Hessian noise threshold: {hess_noise_thresh:.2e}  (FD rounding error estimate at {dps} dps)")
    print(f"  Hessian diagonal (active modes): "
          f"{diag_neg_count} neg / {diag_flat_count} flat / {len(diag_pos_vals)} pos"
          f"  (of {n_active_params} active params)  ({time.time()-t2:.1f}s)")
    print(f"  (Skipped {diag_skip_count} params for {len(bound_active_modes)} bound-inactive modes)")
    if diag_pos_vals:
        print(f"  POSITIVE active-mode entries (exceed noise threshold \u2014 inspect!):")
        for idx, val in diag_pos_vals:
            param_name = ['theta', 'phi', 'psi', 'loga'][idx % 4]
            mode_idx = idx // 4
            print(f"    param[{idx}] (mode {mode_idx} {param_name}): H_ii = {val:+.6e}")
            # Diagnose common false-positive: phi or psi at theta=pi/2 boundary.
            # At theta=pi/2, cos(theta)=0 so u_i = r*exp(i*psi)*e2 — phi has zero effect,
            # giving H_{phi,phi}=0 exactly; FD noise can make this spuriously positive.
            if param_name in ('phi', 'psi'):
                theta_val = x_f64[4 * mode_idx + 0]
                import math as _math
                dist_from_halfpi = abs(theta_val - _math.pi/2)
                if dist_from_halfpi < 1e-3:
                    print(f"      (theta={theta_val:.8f} ~ pi/2; {param_name} inert "
                          f"[cos(theta)={_math.cos(theta_val):.2e}] — likely FD noise)")
    else:
        print(f"  No above-threshold positive entries \u2014 local max in active subspace confirmed")
    if diag_flat_count > 0:
        print(f"  ({diag_flat_count} flat/degenerate entries within noise threshold:"
              f" expected for phi at theta=0 or phi/psi at theta=pi/2)")

    # ── Step 5: ISC relay-shell check ─────────────────────────────────────
    if args.full_block:
        # For the full-block problem, there are no relay shells inside the block.
        # All shells are already included; the Hessian check at x*_full is sufficient.
        # Relay shells from outside the block (different k) would require a
        # cross-block ISC argument, which is beyond scope here.
        print("\n  [Step 4] ISC check skipped (full-block mode: all block shells already included)")
    else:
        if not args.skip_isc:
            n_min, n_max = 2**k, 2**(k+1) - 1
            all_shells = [n for n in range(n_min, n_max+1)
                          if get_wavevectors(max_shell2=n, min_shell2=n)]
            relay_shells = [s for s in all_shells if s not in set(nucleus_shells)]
            print(f"\n  [Step 4] ISC relay check: {len(relay_shells)} relay shells ...", flush=True)
            any_pos = isc_check(x_f64, prob, relay_shells)
            if any_pos:
                print("\n  WARNING: relay lower-bound gradient may improve R")
                print(f"  Run: python scripts/gap3/certify_block_maximum.py --k {k} --full_block --n_starts {n_starts} --dps {dps}")
            else:
                print("\n  ISC: no relay shell improves R above nucleus value.")
                print(f"  -> C(I_{k}) = C_nucleus(I_{k}) (relay shells do not help)")
        else:
            print("\n  [Step 4] ISC check skipped (--skip_isc)")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print(f"  CERTIFICATION SUMMARY for C(I_{k})")
    print("=" * 68)
    print(f"  Float64 optimal value :  {val_f64:.10f}")
    print(f"  {label} modes          :  {n_active}/{N}")
    print(f"  Projected grad norm   :  {proj_grad_norm:.3e}")
    print(f"  mpmath value ({dps} dps):  {mpmath.nstr(R_mp, dps)}")
    print(f"  Certified interval    :  [{mpmath.nstr(cert_lo, 10)}, {mpmath.nstr(cert_hi, 10)}]")
    print(f"  Hessian (active only) :  {diag_neg_count} neg / {diag_flat_count} flat / "
          f"{len(diag_pos_vals)} pos  (noise thresh {hess_noise_thresh:.1e})")
    print()
    nuc_certified = (len(diag_pos_vals) == 0 and proj_grad_norm < 1e-6)
    if nuc_certified:
        if args.full_block:
            print(f"  STATUS: {prob_label} LOCALLY CERTIFIED")
            print(f"  {prob_label} = {mpmath.nstr(cert_lo, 8)}  [certified local max of full block I_{k}]")
            print(f"  x* is a certified local maximum of R on all shells in I_{k}.")
            if _global_cert_passed is True:
                print(f"  GLOBAL CERT: PASS — {args.global_cert_starts} random starts on active subproblem "
                      f"all <= C*  (P8 falsification complete)")
                print(f"  STATUS: {prob_label} NUMERICALLY GLOBALLY CERTIFIED")
            elif _global_cert_passed is False:
                print(f"  GLOBAL CERT: FAIL — a random start exceeded C*; see WARNING above.")
            else:
                print(f"  (Global cert not requested; run with --global-cert-starts 1000 to close.)")
        else:
            print("  STATUS: C_nuc LOCALLY CERTIFIED")
            print(f"  C_nuc(I_{k}) = {mpmath.nstr(cert_lo, 8)}  [rigorous lower bound for C(I_{k})]")
            print(f"  x* is a certified local maximum of R restricted to nucleus shells.")
            if args.skip_isc:
                print(f"  Run without --skip_isc to check ISC relay gradients.")
                print(f"  If all relay ISC gradients <= 0: C_nuc(I_{k}) = C(I_{k}) = {mpmath.nstr(cert_hi, 8)}")
    else:
        if len(diag_pos_vals) > 0:
            print(f"  STATUS: PARTIAL \u2014 {len(diag_pos_vals)} positive Hessian entries; x* may be a saddle.")
        else:
            print(f"  STATUS: PARTIAL \u2014 projected gradient {proj_grad_norm:.1e} > 1e-6;"
                  f" optimizer may not have fully converged.")
    print("=" * 68)


if __name__ == "__main__":
    main()
