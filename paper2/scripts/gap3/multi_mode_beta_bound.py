"""
multi_mode_beta_bound.py
========================
Triad enumeration and coupling-constant optimisation for the
Navier-Stokes enstrophy ratio on T³.

Background
----------
The trilinear form B(u,u,Δu) is the numerator of the scale-invariant
coupling ratio

    R(v) = B(v,v,Δv) / (X(v)² D(v)),   X = ‖∇v‖, D = ‖Δv‖.

Its Fourier expansion is a sum over resonant triples (ell, r, s) with
ell + r + s = 0 (wavevector space), where the Laplacian weight |ell|²
sits on the ell slot:

    B_t = -Im[ |ell_t|² · (s_t · û_{r_t}) · (û*_{ell_t} · û_{s_t}) ].

Algebraic vanishing — triads eliminated at build time
-----------------------------------------------------
One exact cancellation allows an entire triad class to be omitted from
the cache with zero approximation error:

  Same-shell (SS):
    For any div-free u, triads with |ell|² = |r|² = |s|² sum to zero
    exactly.  (Lions antisymmetry applied with equal weights on a single
    shell; Prop 2.4 of ns_cancellation.tex.)

The tempting stronger filter |ell|² = |r|² is not valid for this ordered
Fourier numerator: the upper-lower-upper class can have nonzero aggregate
contribution at a general field.  It must remain in the cache.

Triad array format
------------------
Five parallel arrays describe T filtered triads:

  ell_idx : int32  (T,)    index of ell in the 2N mode list
  ell2    : float64 (T,)   |ell|², the Laplacian weight for this triad
  r_idx   : int32  (T,)    index of r in the 2N mode list
  s_idx   : int32  (T,)    index of s in the 2N mode list
  s_mat   : float64 (T, 3) s as a float vector (for the dot product s·û_r)

The 2N mode list has positive-representative modes at indices [0, N) and
their complex-conjugate negatives at [N, 2N).  Accordingly:

    u_raw[j] = conj(u_raw[j - N])   for j ≥ N.

Cache versioning
----------------
TRIAD_FILTER_VERSION = 2 records that only same-shell filtering is applied.
Cached triad files store this version in meta.npz; any cache built with
a different version is automatically rebuilt.
"""

import numpy as np
from scipy.optimize import minimize
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


# Version tag stored in every triad cache.  Increment when the filter
# logic changes so stale caches are automatically detected and rebuilt.
TRIAD_FILTER_VERSION = 2


# ---------------------------------------------------------------------------
# Divergence-free polarisation basis
# ---------------------------------------------------------------------------

def divfree_basis(k):
    """Return two orthonormal vectors (e1, e2) spanning the div-free plane at k.

    Both e1 and e2 are real, unit-length, perpendicular to k, and mutually
    perpendicular.  Together they span the two-dimensional subspace of
    Fourier coefficients that satisfy the solenoidality condition k·û(k)=0.
    """
    k = np.array(k, dtype=float)
    khat = k / np.linalg.norm(k)
    for trial in [np.array([1., 0., 0.]),
                  np.array([0., 1., 0.]),
                  np.array([0., 0., 1.])]:
        e = trial - (trial @ khat) * khat
        n = np.linalg.norm(e)
        if n > 1e-12:
            e1 = e / n
            break
    e2 = np.cross(khat, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2


# ---------------------------------------------------------------------------
# Wavevector enumeration
# ---------------------------------------------------------------------------

def get_wavevectors(max_shell2, min_shell2=2):
    """Return all positive-representative wavevectors with |k|² ∈ [min_shell2, max_shell2].

    "Positive representative" means we keep exactly one of (k, -k): the
    lexicographically larger one.  This halves the mode count while still
    allowing negative modes to be reconstructed via the reality condition
    û(-k) = conj(û(k)).

    Parameters
    ----------
    max_shell2 : int
        Maximum squared wavenumber (inclusive).
    min_shell2 : int, optional
        Minimum squared wavenumber (default 2).  Shell |k|²=1 contributes
        zero to the denominator D² - λ₁X² and must be excluded from the
        coupling-constant optimisation.

    Returns
    -------
    list of (int, int, int)
        Wavevectors in lexicographic order.

    Notes
    -----
    The enumeration is O(max_shell2) in the double loop: for each (kx, ky)
    pair the valid kz range is computed analytically, giving roughly 8×
    fewer iterations than a naive triple loop over [-R, R]³.
    """
    import math
    pos = []
    R = math.isqrt(max_shell2)
    for kx in range(-R, R + 1):
        kx2 = kx * kx
        if kx2 > max_shell2:
            continue
        lim_y = math.isqrt(max_shell2 - kx2)
        for ky in range(-lim_y, lim_y + 1):
            ky2 = ky * ky
            rem = max_shell2 - kx2 - ky2
            if rem < 0:
                continue
            lo2 = max(0, min_shell2 - kx2 - ky2)
            lo = math.isqrt(lo2)
            if lo * lo < lo2:
                lo += 1
            hi = math.isqrt(rem)
            for kz in range(lo, hi + 1):
                k2 = kx2 + ky2 + kz * kz
                if k2 < min_shell2 or k2 > max_shell2:
                    continue
                for kz_val in ([0] if kz == 0 else [kz, -kz]):
                    k = (kx, ky, kz_val)
                    neg_k = (-kx, -ky, -kz_val)
                    if k > neg_k:
                        pos.append(k)
    return pos


# ---------------------------------------------------------------------------
# Triad enumeration helpers
# ---------------------------------------------------------------------------

def _is_same_shell(ell2, r2, s2):
    """Return True if the triad (ell, r, s=ell-r) must be dropped.

    Only the same-shell class |ell|² = |r|² = |s|² is removed.  The broader
    |ell|² = |r|² class is not a build-time cancellation for this ordered
    numerator.
    """
    return ell2 == r2 == s2


# ---------------------------------------------------------------------------
# In-memory triad enumeration (small problems)
# ---------------------------------------------------------------------------

def precompute_triads(wavevecs, progress_label=None, progress_every=0):
    """Enumerate all resonant triads for a set of wavevectors, in memory.

    Builds the full 2N mode list (positive reps + their negatives) and
    enumerates all ordered triples (ell, r, s) with ell = r + s where
    all three modes are in the 2N list.  Same-shell triads are silently
    dropped (see module docstring).

    This function keeps everything in Python lists and is suited for small
    problems (N ≲ 1000 modes, T ≲ a few million triads).  For large full-
    block runs use precompute_triads_memmap instead.

    Parameters
    ----------
    wavevecs : list of (int, int, int)
        Positive-representative wavevectors (output of get_wavevectors).
    progress_label : str, optional
        If given, print progress lines every progress_every ell iterations.
    progress_every : int, optional
        Print interval in ell-loop iterations (0 = no progress).

    Returns
    -------
    N : int
        Number of positive-representative modes.
    ell_idx : np.ndarray, shape (T,), int32
        Index of the ell (Δ-output) mode in the 2N all-wv list.
    ell2_arr : np.ndarray, shape (T,), float64
        |ell|² for each triad (the Laplacian weight).
    r_idx : np.ndarray, shape (T,), int32
        Index of the r (advecting input) mode.
    s_idx : np.ndarray, shape (T,), int32
        Index of the s (transported input) mode.
    s_mat : np.ndarray, shape (T, 3), float64
        s vector as floats (used in the dot product s·û_r).
    """
    import time

    N = len(wavevecs)
    pos_list = list(wavevecs)
    neg_list = [tuple(-ki for ki in k) for k in pos_list]
    all_wv   = pos_list + neg_list
    wv_index = {k: i for i, k in enumerate(all_wv)}
    wv_sq    = {k: sum(ki * ki for ki in k) for k in all_wv}

    ell_idx_list, ell2_list = [], []
    r_idx_list, s_idx_list, s_list = [], [], []

    progress_every = int(progress_every or 0)
    t0    = time.time()
    n_all = len(all_wv)

    for ei, ell in enumerate(all_wv):
        ell2 = wv_sq[ell]
        if ell2 == 0:
            continue
        for ri, r in enumerate(all_wv):
            s = (ell[0] - r[0], ell[1] - r[1], ell[2] - r[2])
            if s not in wv_index:
                continue
            if _is_same_shell(ell2, wv_sq[r], wv_sq[s]):
                continue
            ell_idx_list.append(ei)
            ell2_list.append(ell2)
            r_idx_list.append(ri)
            s_idx_list.append(wv_index[s])
            s_list.append(s)
        if (progress_label is not None and progress_every > 0 and
                ((ei + 1) % progress_every == 0 or ei + 1 == n_all)):
            print(f"  [precompute_triads] {progress_label}:"
                  f" ell {ei + 1:,}/{n_all:,}, triads={len(ell_idx_list):,}"
                  f" ({time.time() - t0:.1f}s)", flush=True)

    return (N,
            np.array(ell_idx_list, dtype=np.int32),
            np.array(ell2_list,    dtype=np.float64),
            np.array(r_idx_list,   dtype=np.int32),
            np.array(s_idx_list,   dtype=np.int32),
            np.array(s_list,       dtype=np.float64))


# ---------------------------------------------------------------------------
# Memmap triad cache (large full-block problems)
# ---------------------------------------------------------------------------

def precompute_triads_memmap(wavevecs, cache_dir, progress_label=None,
                             progress_every=0, reuse=True):
    """Enumerate resonant triads into memory-mapped .npy files on disk.

    For large full-block runs (k ≥ 6, T ≥ tens of millions of triads) the
    in-memory path spends tens of GB on Python lists before NumPy arrays
    even exist.  This function writes triads directly into pre-allocated
    memmap files, keeping peak Python-heap usage small.

    Same-shell triads are filtered out, reducing the cache size without any
    loss of correctness.  See the module docstring for the mathematical
    justification.

    Cache reuse
    -----------
    If reuse=True and cache_dir already contains valid files (matching N,
    wavevec fingerprint, and TRIAD_FILTER_VERSION), the files are reopened
    in mmap_mode='r+' and returned immediately without recomputing.  Any
    mismatch (including the filter-version check) triggers a full rebuild.

    Two-pass algorithm
    ------------------
    Pass 1 (count): iterate over all (ell, r) pairs to count T, applying
    the SS/ULU filter, so that the memmap shape can be allocated exactly.
    Pass 2 (fill): iterate again to write the filtered triads.

    Parameters
    ----------
    wavevecs : list of (int, int, int)
        Positive-representative wavevectors.
    cache_dir : str
        Directory for the five .npy memmap files and meta.npz.
    progress_label : str, optional
        Tag for progress prints (e.g. 'full I_8').
    progress_every : int, optional
        Print progress every this many ell iterations.
    reuse : bool, optional
        Return cached files if valid (default True).

    Returns
    -------
    Same six-tuple as precompute_triads (N, ell_idx, ell2, r_idx, s_idx, s_mat),
    but the arrays are backed by memory-mapped files in cache_dir.
    """
    import time

    os.makedirs(cache_dir, exist_ok=True)
    wavevec_arr = np.array(wavevecs, dtype=np.int16)
    meta_path   = os.path.join(cache_dir, 'meta.npz')
    paths = {
        'ell_idx': os.path.join(cache_dir, 'ell_idx.npy'),
        'ell2':    os.path.join(cache_dir, 'ell2.npy'),
        'r_idx':   os.path.join(cache_dir, 'r_idx.npy'),
        's_idx':   os.path.join(cache_dir, 's_idx.npy'),
        's_mat':   os.path.join(cache_dir, 's_mat.npy'),
    }

    # --- Cache validation ---
    if reuse and os.path.isfile(meta_path) and all(os.path.isfile(p)
                                                    for p in paths.values()):
        try:
            meta = np.load(meta_path)
            cached_version = int(meta.get('filter_version', 0))
            n_ok     = int(meta['N']) == len(wavevecs)
            wv_ok    = np.array_equal(meta['wavevecs'], wavevec_arr)
            if n_ok and wv_ok:
                T = int(meta['T'])
                cached_arrays = {
                    name: np.load(path, mmap_mode='r+')
                    for name, path in paths.items()
                }
                shape_ok = (
                    cached_arrays['ell_idx'].shape == (T,) and
                    cached_arrays['ell2'].shape == (T,) and
                    cached_arrays['r_idx'].shape == (T,) and
                    cached_arrays['s_idx'].shape == (T,) and
                    cached_arrays['s_mat'].shape == (T, 3)
                )
                if not shape_ok:
                    shape_msg = ', '.join(
                        f"{name}{arr.shape}" for name, arr in cached_arrays.items())
                    print(f"  [precompute_triads] cache shape mismatch for"
                          f" {progress_label or 'triads'}: meta T={T:,},"
                          f" arrays={shape_msg}; rebuilding ...", flush=True)
                elif cached_version != TRIAD_FILTER_VERSION:
                    print(f"  [precompute_triads] cache filter_version="
                          f"{cached_version} != {TRIAD_FILTER_VERSION};"
                          f" rebuilding ...", flush=True)
                else:
                    print(f"  [precompute_triads] using cached"
                          f" {progress_label or 'triads'}:"
                          f" {len(wavevecs)} modes, {T:,} triads"
                          f" (filter_version={cached_version})"
                          f" -> {cache_dir}", flush=True)
                    return (len(wavevecs),
                                                        cached_arrays['ell_idx'],
                                                        cached_arrays['ell2'],
                                                        cached_arrays['r_idx'],
                                                        cached_arrays['s_idx'],
                                                        cached_arrays['s_mat'])
            else:
                print(f"  [precompute_triads] cache stale (N/wavevec mismatch),"
                      f" rebuilding ...", flush=True)
        except Exception as exc:
            print(f"  [precompute_triads] cache unreadable ({exc!r}),"
                  f" rebuilding ...", flush=True)

    N        = len(wavevecs)
    pos_list = list(wavevecs)
    neg_list = [tuple(-ki for ki in k) for k in pos_list]
    all_wv   = pos_list + neg_list
    wv_index = {k: i for i, k in enumerate(all_wv)}
    wv_sq    = {k: sum(ki * ki for ki in k) for k in all_wv}
    n_all    = len(all_wv)
    progress_every = int(progress_every or 0)

    def _should_print(ei):
        return (progress_label is not None and progress_every > 0
                and ((ei + 1) % progress_every == 0 or ei + 1 == n_all))

    # --- Pass 1: count filtered triads ---
    t0 = time.time()
    T  = 0
    for ei, ell in enumerate(all_wv):
        ell2 = wv_sq[ell]
        if ell2 == 0:
            continue
        for r in all_wv:
            s = (ell[0] - r[0], ell[1] - r[1], ell[2] - r[2])
            if s in wv_index:
                if _is_same_shell(ell2, wv_sq[r], wv_sq[s]):
                    continue
                T += 1
        if _should_print(ei):
            print(f"  [precompute_triads] {progress_label} count:"
                  f" ell {ei + 1:,}/{n_all:,}, triads={T:,}"
                  f" ({time.time() - t0:.1f}s)", flush=True)

    print(f"  [precompute_triads] {progress_label or 'triads'}:"
          f" allocating {T:,} filtered triads in {cache_dir}", flush=True)

    # --- Allocate memmap files ---
    ell_idx  = np.lib.format.open_memmap(paths['ell_idx'], mode='w+',
                                         dtype=np.int32,   shape=(T,))
    ell2_arr = np.lib.format.open_memmap(paths['ell2'],    mode='w+',
                                         dtype=np.float64, shape=(T,))
    r_idx    = np.lib.format.open_memmap(paths['r_idx'],   mode='w+',
                                         dtype=np.int32,   shape=(T,))
    s_idx    = np.lib.format.open_memmap(paths['s_idx'],   mode='w+',
                                         dtype=np.int32,   shape=(T,))
    s_mat    = np.lib.format.open_memmap(paths['s_mat'],   mode='w+',
                                         dtype=np.float64, shape=(T, 3))

    # --- Pass 2: fill filtered triads ---
    t1  = time.time()
    pos = 0
    for ei, ell in enumerate(all_wv):
        ell2 = wv_sq[ell]
        if ell2 == 0:
            continue
        for ri, r in enumerate(all_wv):
            s  = (ell[0] - r[0], ell[1] - r[1], ell[2] - r[2])
            si = wv_index.get(s)
            if si is None:
                continue
            if _is_same_shell(ell2, wv_sq[r], wv_sq[s]):
                continue
            ell_idx[pos]    = ei
            ell2_arr[pos]   = ell2
            r_idx[pos]      = ri
            s_idx[pos]      = si
            s_mat[pos, 0]   = s[0]
            s_mat[pos, 1]   = s[1]
            s_mat[pos, 2]   = s[2]
            pos += 1
        if _should_print(ei):
            print(f"  [precompute_triads] {progress_label} fill:"
                  f" ell {ei + 1:,}/{n_all:,},"
                  f" triads={pos:,}/{T:,}"
                  f" ({time.time() - t1:.1f}s fill,"
                  f" {time.time() - t0:.1f}s total)", flush=True)

    for arr in (ell_idx, ell2_arr, r_idx, s_idx, s_mat):
        arr.flush()
    np.savez(meta_path,
             N=np.int64(N), T=np.int64(T),
             wavevecs=wavevec_arr,
             filter_version=np.int64(TRIAD_FILTER_VERSION))
    print(f"  [precompute_triads] {progress_label or 'triads'} cache ready:"
            f" {T:,} triads (same-shell filtered, version={TRIAD_FILTER_VERSION})",
          flush=True)
    return (N,
            np.load(paths['ell_idx'], mmap_mode='r+'),
            np.load(paths['ell2'],    mmap_mode='r+'),
            np.load(paths['r_idx'],   mmap_mode='r+'),
            np.load(paths['s_idx'],   mmap_mode='r+'),
            np.load(paths['s_mat'],   mmap_mode='r+'))


# ---------------------------------------------------------------------------
# Mode-vector builder
# ---------------------------------------------------------------------------

def params_to_u(params, N, e1s, e2s):
    """Build the (2N, 3) complex mode-vector array from angle parameters.

    Each positive mode i has two divergence-free degrees of freedom
    (θᵢ, φᵢ, ψᵢ):

        û_pos[i] = cos(θ) exp(iφ) e1[i] + sin(θ) exp(iψ) e2[i]

    The negative-mode half is the complex conjugate (reality condition).

    Parameters
    ----------
    params : np.ndarray, shape (3N,)
        Angles packed as [θ₀, φ₀, ψ₀, θ₁, φ₁, ψ₁, ...].
    N : int
        Number of positive-representative modes.
    e1s, e2s : np.ndarray, shape (N, 3)
        Divergence-free basis vectors for each mode.

    Returns
    -------
    np.ndarray, shape (2N, 3), complex
        u_raw[0:N] = positive modes, u_raw[N:2N] = conjugate negatives.
    """
    theta = params[0::3][:, None]
    phi   = params[1::3][:, None]
    psi   = params[2::3][:, None]
    u_pos = (np.cos(theta) * np.exp(1j * phi) * e1s
             + np.sin(theta) * np.exp(1j * psi) * e2s)   # (N, 3)
    return np.vstack([u_pos, u_pos.conj()])               # (2N, 3)


# ---------------------------------------------------------------------------
# Vectorised trilinear form
# ---------------------------------------------------------------------------

def compute_B_vec(u, ell_idx, ell2, r_idx, s_idx, s_mat):
    """Evaluate B(u, u, Δu) from pre-enumerated triad index arrays.

    Uses the representation

        B = -Im Σ_t |ell_t|² · (s_t · u[r_t]) · (u[ell_t]* · u[s_t])

    where the sum is over all filtered triads (SS and ULU already removed).

    Parameters
    ----------
    u : np.ndarray, shape (2N, 3), complex
        Mode vectors (build with params_to_u or the certify_* builders).
    ell_idx, r_idx, s_idx : np.ndarray, (T,), int32
        Triad index arrays from precompute_triads[_memmap].
    ell2 : np.ndarray, (T,), float64
        Laplacian weights |ell|².
    s_mat : np.ndarray, (T, 3), float64
        s vectors for the dot product s·û_r.

    Returns
    -------
    float
        The scalar value B(u, u, Δu).
    """
    sdu = np.einsum('td,td->t', s_mat, u[r_idx])              # s·u_r
    ced = np.einsum('td,td->t', u[ell_idx].conj(), u[s_idx])  # u_ell*·u_s
    return -np.imag(np.dot(ell2 * sdu, ced))


# ---------------------------------------------------------------------------
# Fixed-Δ denominator (β optimisation)
# ---------------------------------------------------------------------------

def compute_Delta(wavevecs):
    """Compute the fixed denominator Δ = Σ_k 2|k|²(|k|²-1) for unit amplitudes.

    This is the denominator of β = B/Δ in the early unit-amplitude
    formulation; it is not used by the current amplitude-varying R = B/(X²D)
    objective but is retained for the β-scan entry point.
    """
    return sum(2 * sum(ki * ki for ki in k) * (sum(ki * ki for ki in k) - 1)
               for k in wavevecs)


# ---------------------------------------------------------------------------
# β objective and worker (legacy unit-amplitude formulation)
# ---------------------------------------------------------------------------

def neg_beta(params, N, e1s, e2s, ell_idx, ell2, r_idx, s_idx, s_mat, Delta):
    """Return -β for minimisation; β = B(u,u,Δu) / Δ with unit amplitudes."""
    u = params_to_u(params, N, e1s, e2s)
    B = compute_B_vec(u, ell_idx, ell2, r_idx, s_idx, s_mat)
    return -B / Delta


def _restart_worker(seed, n_local, N, e1s, e2s,
                    ell_idx, ell2, r_idx, s_idx, s_mat, Delta, bounds):
    """Run n_local random L-BFGS-B restarts and return (best_beta, best_params).

    Module-level so it is picklable by ProcessPoolExecutor.
    """
    rng     = np.random.default_rng(seed)
    lo      = np.array([b[0] for b in bounds])
    hi      = np.array([b[1] for b in bounds])
    args    = (N, e1s, e2s, ell_idx, ell2, r_idx, s_idx, s_mat, Delta)
    best_beta = -np.inf
    best_p    = None
    for _ in range(n_local):
        p0  = rng.uniform(lo, hi)
        res = minimize(neg_beta, p0, args=args, method='L-BFGS-B', bounds=bounds,
                       options={'ftol': 1e-13, 'gtol': 1e-10, 'maxiter': 500})
        b = -res.fun
        if b > best_beta:
            best_beta = b
            best_p    = res.x
    return best_beta, best_p


# ---------------------------------------------------------------------------
# High-level optimisation entry points
# ---------------------------------------------------------------------------

def find_max_beta(max_shell2=2, n_starts=1000, verbose=True, return_params=False,
                  n_workers=None, warm_start=None):
    """Find the maximum of β = B/Δ over divergence-free unit-amplitude fields.

    Uses multi-start L-BFGS-B optimisation in parallel.  This is the legacy
    entry point for the fixed-Δ β bound; the current paper uses the
    amplitude-varying ratio R = B/(X²D) optimised by certify_block_maximum_gpu.py.

    Parameters
    ----------
    max_shell2 : int
        Include all shells with |k|² ≤ max_shell2.
    n_starts : int
        Total number of random restarts.
    verbose : bool
        Print progress and result.
    return_params : bool
        If True, return (best_beta, params) rather than just best_beta.
    n_workers : int or None
        CPU workers for parallelism (default: all cores).
    warm_start : np.ndarray or None
        Optional initial parameter vector; polished first if provided.

    Returns
    -------
    float or (float, np.ndarray)
        Best β found, optionally with the parameter vector at the optimum.
    """
    wavevecs = get_wavevectors(max_shell2)
    N        = len(wavevecs)
    Delta    = compute_Delta(wavevecs)

    if verbose:
        all_shells = sorted(set(sum(ki * ki for ki in k) for k in wavevecs))
        print(f'  Active shells:  {all_shells}')
        print(f'  Positive modes: {N}   (params: {3 * N})')
        print(f'  Δ = {Delta}')

    if Delta == 0:
        if verbose:
            print('  Δ=0.  β=0 trivially.')
        empty = np.zeros(3 * N)
        return (0.0, empty) if return_params else 0.0

    e1s = np.array([divfree_basis(k)[0] for k in wavevecs])
    e2s = np.array([divfree_basis(k)[1] for k in wavevecs])
    N_, ell_idx, ell2, r_idx, s_idx, s_mat = precompute_triads(wavevecs)

    n_workers = n_workers or os.cpu_count() or 1
    n_workers = min(n_workers, n_starts)

    if verbose:
        print(f'  Active triads:  {len(ell_idx):,}')
        print(f'  Random-restart L-BFGS-B'
              f' ({n_starts} starts, {n_workers} workers) ...')

    bounds = [(0, np.pi / 2), (0, 2 * np.pi), (0, 2 * np.pi)] * N
    base, rem = divmod(n_starts, n_workers)
    n_per_worker = [base + (1 if i < rem else 0) for i in range(n_workers)]
    seeds = [42 + i * 1000 for i in range(n_workers)]

    best_beta = -np.inf
    best_p    = None

    if warm_start is not None and len(warm_start) == 3 * N:
        args_ws = (N, e1s, e2s, ell_idx, ell2, r_idx, s_idx, s_mat, Delta)
        res_ws  = minimize(neg_beta, warm_start, args=args_ws,
                           method='L-BFGS-B', bounds=bounds,
                           options={'ftol': 1e-14, 'gtol': 1e-11, 'maxiter': 2000})
        b_ws = -res_ws.fun
        if b_ws > best_beta:
            best_beta = b_ws
            best_p    = res_ws.x
        if verbose:
            print(f'  Warm-start polish: β = {b_ws:.8f}')

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [
            pool.submit(_restart_worker, seeds[i], n_per_worker[i],
                        N, e1s, e2s, ell_idx, ell2, r_idx, s_idx, s_mat,
                        Delta, bounds)
            for i in range(n_workers)
        ]
        for fut in as_completed(futures):
            b, p = fut.result()
            if b > best_beta:
                best_beta = b
                best_p    = p

    args = (N, e1s, e2s, ell_idx, ell2, r_idx, s_idx, s_mat, Delta)
    res  = minimize(neg_beta, best_p, args=args, method='L-BFGS-B', bounds=bounds,
                    options={'ftol': 1e-16, 'gtol': 1e-13, 'maxiter': 10000})
    best_beta = -res.fun

    if verbose:
        u = params_to_u(res.x, N, e1s, e2s)
        B = compute_B_vec(u, ell_idx, ell2, r_idx, s_idx, s_mat)
        print(f'  Max β = {best_beta:.8f}   (B = {B:.4f},  Δ = {Delta})')

    return (best_beta, res.x) if return_params else best_beta


def scan_shells(shell_range, n_starts=1000, extremizer_path=None, n_workers=None):
    """Scan β = B/Δ over a range of max_shell² values.

    Runs find_max_beta for each value in shell_range, using the previous
    extremizer as a warm start for the next shell.  Prints a summary table
    and optionally writes extremizer parameters to a JSON file.

    Parameters
    ----------
    shell_range : iterable of int
        Sequence of max_shell² values (e.g. range(2, 20)).
    n_starts : int
        Random restarts per shell.
    extremizer_path : str or None
        If given, write extremizer params and wavevectors to this JSON file.
    n_workers : int or None
        CPU workers for parallelism.
    """
    print('╔══════════════════════════════════════════════════════════════╗')
    print('║  MULTI-MODE β BOUND SCAN  (unit amplitude per mode)         ║')
    print('╚══════════════════════════════════════════════════════════════╝')
    results        = []
    prev_beta      = None
    prev_params    = None
    prev_wavevecs  = None

    for s2 in shell_range:
        print(f'\n{"─" * 60}')
        print(f'  max_shell² = {s2}')

        warm          = None
        cur_wavevecs  = get_wavevectors(s2)
        if prev_params is not None and prev_wavevecs is not None:
            N_prev = len(prev_wavevecs)
            N_cur  = len(cur_wavevecs)
            if N_cur > N_prev:
                warm = np.zeros(3 * N_cur)
                warm[:3 * N_prev] = prev_params
            elif N_cur == N_prev:
                warm = prev_params.copy()

        b, params = find_max_beta(max_shell2=s2, n_starts=n_starts,
                                   return_params=True, n_workers=n_workers,
                                   warm_start=warm)
        incr = ('' if prev_beta is None
                else f'   Δβ = {b - prev_beta:+.6f}')
        results.append((s2, b, params))
        print(f'  → max β = {b:.8f}'
              f'   {"✓" if b <= 0.5 + 1e-6 else "✗ EXCEEDS 1/2!"}{incr}')
        prev_beta     = b
        prev_params   = params
        prev_wavevecs = cur_wavevecs

    print('\n' + '═' * 60)
    print('SUMMARY')
    print('─' * 60)
    print(f'  {"S":>4}   {"max β":>12}   {"Δβ":>10}   status')
    print('  ' + '─' * 50)
    prev = None
    for s2, b, _ in results:
        incr   = '' if prev is None else f'{b - prev:+.6f}'
        status = 'PROVED ✓' if b <= 0.5 + 1e-6 else 'FAILED ✗'
        print(f'  {s2:>4}   {b:>12.8f}   {incr:>10}   [{status}]')
        prev = b
    print()
    if all(b <= 0.5 + 1e-6 for _, b, _ in results):
        print('All cases confirm β ≤ 1/2.  ✓')
    else:
        print('WARNING: some cases exceed 1/2!')

    if extremizer_path is not None:
        import json
        data = {}
        for s2, b, params in results:
            wv = get_wavevectors(s2)
            data[str(s2)] = {
                'max_beta':    float(b),
                'wavevectors': [list(k) for k in wv],
                'params':      params.tolist(),
            }
        with open(extremizer_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f'Extremizers written to {extremizer_path}')

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='β-bound scan for the NS enstrophy coupling constant.')
    parser.add_argument('--max_shell2', type=int, default=2,
                        help='Maximum |k|² to include (default 2).')
    parser.add_argument('--min_shell2', type=int, default=2,
                        help='Minimum |k|² to include (default 2; '
                             'shell-1 modes have zero Δ-weight and must be excluded).')
    parser.add_argument('--scan', action='store_true',
                        help='Scan all shells from min_shell2 to max_shell2.')
    parser.add_argument('--n_starts', type=int, default=1000,
                        help='Random L-BFGS-B restarts per shell (default 1000).')
    parser.add_argument('--n_workers', type=int, default=None,
                        help='CPU worker processes (default: all cores).')
    parser.add_argument('--save_extremizers', default=None, metavar='PATH',
                        help='Write extremizer parameters to this JSON file.')
    args = parser.parse_args()

    if args.scan:
        scan_shells(range(args.min_shell2, args.max_shell2 + 1),
                    n_starts=args.n_starts,
                    extremizer_path=args.save_extremizers,
                    n_workers=args.n_workers)
    else:
        find_max_beta(max_shell2=args.max_shell2,
                      n_starts=args.n_starts,
                      n_workers=args.n_workers)
