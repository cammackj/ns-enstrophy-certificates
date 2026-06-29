"""
max_b_over_keff.py
==================
Find the maximum of

    C = B(u,u,Δu) / (X² · D)

over ALL divergence-free fields u on T³, where:
    B(u,u,Δu) — the NS trilinear form (with Leray projector)
    X² = Σ_k k²|û_k|²   (H¹ energy)
    D  = sqrt(Σ_k k⁴|û_k|²)  (H² semi-norm)

This is the direct adversarial test for Conjecture 15.4: the correct measure
of worst-case cancellation in the NS nonlinearity.  Unlike lp_decomp_bound.py,
which only probes isotropic spectral-slope families, this optimizer is free to
concentrate amplitude on any subset of shells and any polarisation — finding
the true worst-case field configuration for any given triad set.

NOTE on notation: The module was originally described using K_eff = D/X, so that
"B/(K_eff·X²)" = "B/(X²·D/X)" = "B·X/(X²·D)" = "B/(X·D)".  That is NOT the
same as B/(X²·D).  The code has always computed B/(X²·D) correctly; only the
prose description was ambiguous.  The canonical objective is B/(X²·D).

Scale invariance: under u → λu, B ~ λ³, X² ~ λ², D ~ λ, so B/(X²·D) is
unchanged.  Each mode therefore carries an independent log-amplitude parameter
that controls the spectral shape with no normalisation constraint needed.

Parameters per mode: (θ, φ, ψ, log_a), giving 4N free parameters total.

RELATIONSHIP TO gap3_principled_scan.py
----------------------------------------
Both scripts optimise B/(X²·D) via `neg_ratio_and_grad`, but with DIFFERENT
triad sets:

  find_max_ratio / gap3_isc_at_true_optimizer.run_full_block:
    Calls precompute_triads(wvs) where wvs = ALL modes in the requested shell
    range.  This discovers every resonant triad.  The resulting optimizer can
    reach any basin, regardless of which shells it lives on.

  gap3_principled_scan.run_block:
    Builds triads via _build_problem(), which uses a cross-shell ANCHOR SEARCH
    anchored on the SMALLEST shell in I_k.  Only triads that include a mode from
    the smallest shell are found.  Modes that never touch the smallest shell are
    absent, and entire basins not reachable from the anchor are structurally
    invisible.  _restrict_to_active_modes further prunes to the anchor-reachable
    set before calling precompute_triads — so even the final rebuild cannot
    recover shells outside the anchor-reachable component.

  Consequence: run_block may miss the true global maximum.  For k=5,
  run_full_block (ISC script, Apr 5 2026) found C=0.016735 with support on
  {32-36,38,41,43,50,51,54,59} — shells that form a rich mutual-triad network
  but barely couple to the anchor shell 32.  run_block found only 0.015347
  ({32,34,37,41,61}), not because the code was buggy but because the anchor
  search is structurally blind to the higher basin.

Usage
-----
    python scripts/gap3/max_b_over_keff.py --max_shell2 9 --n_starts 500
    python scripts/gap3/max_b_over_keff.py --scan --min_shell2 2 --max_shell2 25 --n_starts 300
"""

import numpy as np
from scipy.optimize import minimize
import argparse, os
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
sys.path.insert(0, '.')
from scripts.gap3.multi_mode_beta_bound import (
    get_wavevectors, divfree_basis, precompute_triads, compute_B_vec
)


# ─── field parameterisation ───────────────────────────────────────────────────

def params_to_u_amp(params, N, e1s, e2s, k2s):
    """
    Build u_raw (2N,3) from params (4N,) and return (u_raw, X², D).

    Layout: [θ₀,φ₀,ψ₀,log_a₀,  θ₁,φ₁,ψ₁,log_a₁, ...]
    |û_i|² = exp(log_a_i)  — free amplitude per mode, no normalisation.
    """
    theta = params[0::4][:, None]   # (N,1)
    phi   = params[1::4][:, None]
    psi   = params[2::4][:, None]
    a     = np.exp(params[3::4])    # (N,)  amplitudes squared

    r = np.sqrt(a)
    u_pos = r[:, None] * (np.cos(theta) * np.exp(1j * phi) * e1s
                         + np.sin(theta) * np.exp(1j * psi) * e2s)  # (N,3)
    u_raw = np.vstack([u_pos, u_pos.conj()])                         # (2N,3)

    X2 = 2.0 * float(np.dot(k2s, a))       # X² = 2 Σ k²|û|²
    D2 = 2.0 * float(np.dot(k2s ** 2, a))  # D² = 2 Σ k⁴|û|²
    D  = np.sqrt(max(D2, 1e-60))
    return u_raw, X2, D


# ─── objective + analytic gradient ───────────────────────────────────────────

def _build_u(params, N, e1s, e2s, k2s):
    """Build u_pos, u_raw, a, r, cos_theta, sin_theta, exp_iphi, exp_ipsi, X², D."""
    theta = params[0::4]
    phi   = params[1::4]
    psi   = params[2::4]
    a     = np.exp(params[3::4])
    r     = np.sqrt(a)
    cth, sth  = np.cos(theta), np.sin(theta)
    ephi, epsi = np.exp(1j * phi), np.exp(1j * psi)
    u_pos = r[:, None] * (cth[:, None] * ephi[:, None] * e1s
                         + sth[:, None] * epsi[:, None] * e2s)
    u_raw = np.vstack([u_pos, u_pos.conj()])
    X2 = 2.0 * float(np.dot(k2s, a))
    D2 = 2.0 * float(np.dot(k2s ** 2, a))
    D  = np.sqrt(max(D2, 1e-60))
    return u_pos, u_raw, a, r, cth, sth, ephi, epsi, X2, D


def neg_ratio_and_grad(params, N, e1s, e2s, k2s, ell_idx, ell2, r_idx, s_idx, s_mat):
    """
    Returns (-B/(X²D), gradient) with analytic Jacobian.

    Derivation (Wirtinger chain rule):
      B = -Im(Σ_t ell2_t * (s_t·u[r_t]) * conj(u[ell_t])·u[s_t])
      For each positive mode i, define:
        Gv[j]     += ell2_t * ced_t * s_mat_t     (j = r_t slot)
        Gv[j]     += ell2_t * sdu_t * u[ell_t]*   (j = s_t slot)
        Gvconj[j] += ell2_t * sdu_t * u[s_t]      (j = ell_t slot)
      W[i] = conj(Gvconj[i]) - Gv[i] - Gvconj[N+i] + conj(Gv[N+i])
      ∂B/∂param_k(i) = Im(W[i] · v_{k,i})
    """
    u_pos, u_raw, a, r, cth, sth, ephi, epsi, X2, D = _build_u(params, N, e1s, e2s, k2s)
    XD = X2 * D
    if XD < 1e-60:
        return 0.0, np.zeros(4 * N)

    # forward quantities
    sdu = np.einsum('td,td->t', s_mat, u_raw[r_idx])
    ced = np.einsum('td,td->t', u_raw[ell_idx].conj(), u_raw[s_idx])
    B = float(-np.imag(np.dot(ell2 * sdu, ced)))

    # backward: accumulate Gv (2N,3) and Gvconj (2N,3)
    # np.add.at is unbuffered and slow for large T; np.bincount is vectorized/SIMD.
    Gv     = np.zeros((2 * N, 3), complex)
    Gvconj = np.zeros((2 * N, 3), complex)
    cr = (ell2 * ced)          # (T,)
    cs = (ell2 * sdu)          # (T,)
    N2 = 2 * N

    def _scatter(dest, idx, weights_row):
        """Scatter-add a (T,3) complex array into dest[(2N,3)] at given indices."""
        idx64 = idx.astype(np.int64)
        for d in range(3):
            col = weights_row[:, d]
            dest[:, d].real += np.bincount(idx64, weights=col.real, minlength=N2)
            dest[:, d].imag += np.bincount(idx64, weights=col.imag, minlength=N2)

    _scatter(Gv,     r_idx,   cr[:, None] * s_mat)
    _scatter(Gv,     s_idx,   cs[:, None] * u_raw[ell_idx].conj())
    _scatter(Gvconj, ell_idx, cs[:, None] * u_raw[s_idx])

    # adjoint gradient vector W[i] for i = 0..N-1
    W = Gvconj[:N].conj() - Gv[:N] - Gvconj[N:] + Gv[N:].conj()  # (N,3)

    # variation vectors v(param) for each mode i
    v_th = r[:, None] * (-sth[:, None] * ephi[:, None] * e1s
                         + cth[:, None] * epsi[:, None] * e2s)
    v_ph = r[:, None] * cth[:, None] * (1j * ephi[:, None]) * e1s
    v_ps = r[:, None] * sth[:, None] * (1j * epsi[:, None]) * e2s
    v_la = 0.5 * u_pos   # ∂u_pos/∂log_a = u_pos / 2

    # dB / d(param) = Im(W · v)
    dB_th = np.imag(np.einsum('id,id->i', W, v_th))
    dB_ph = np.imag(np.einsum('id,id->i', W, v_ph))
    dB_ps = np.imag(np.einsum('id,id->i', W, v_ps))
    dB_la = np.imag(np.einsum('id,id->i', W, v_la))

    # ∂(X²D)/∂log_a_i = a_i (2k²_i D + X² k⁴_i / D)
    dXD_la = a * (2.0 * k2s * D + X2 * k2s ** 2 / D)

    # gradient of objective = -B / XD
    inv_XD  = 1.0 / XD
    inv_XD2 = inv_XD * inv_XD
    dobj_th = -dB_th * inv_XD
    dobj_ph = -dB_ph * inv_XD
    dobj_ps = -dB_ps * inv_XD
    dobj_la = -(dB_la * XD - B * dXD_la) * inv_XD2

    grad = np.empty(4 * N)
    grad[0::4] = dobj_th
    grad[1::4] = dobj_ph
    grad[2::4] = dobj_ps
    grad[3::4] = dobj_la
    return -B / XD, grad


def neg_ratio(params, N, e1s, e2s, k2s, ell_idx, ell2, r_idx, s_idx, s_mat):
    """Value-only wrapper (kept for backward compat)."""
    val, _ = neg_ratio_and_grad(params, N, e1s, e2s, k2s,
                                 ell_idx, ell2, r_idx, s_idx, s_mat)
    return val


# ─── parallel worker (top-level for pickling) ─────────────────────────────────

def _worker(seed, n_local, N, e1s, e2s, k2s,
            ell_idx, ell2, r_idx, s_idx, s_mat, bounds):
    rng  = np.random.default_rng(seed)
    lo   = np.array([b[0] for b in bounds])
    hi   = np.array([b[1] for b in bounds])
    args = (N, e1s, e2s, k2s, ell_idx, ell2, r_idx, s_idx, s_mat)
    best_val = -np.inf
    best_p   = None
    for _ in range(n_local):
        p0  = rng.uniform(lo, hi)
        res = minimize(neg_ratio_and_grad, p0, args=args, method='L-BFGS-B',
                       jac=True, bounds=bounds,
                       options={'ftol': 1e-13, 'gtol': 1e-10, 'maxiter': 500})
        v = -res.fun
        if v > best_val:
            best_val = v
            best_p   = res.x
    return best_val, best_p


# ─── main optimiser ───────────────────────────────────────────────────────────

def find_max_ratio(max_shell2=9, n_starts=500, verbose=True, n_workers=None):
    wavevecs = get_wavevectors(max_shell2)
    N  = len(wavevecs)
    k2s = np.array([sum(ki * ki for ki in k) for k in wavevecs], dtype=float)
    e1s = np.array([divfree_basis(k)[0] for k in wavevecs])
    e2s = np.array([divfree_basis(k)[1] for k in wavevecs])

    N_, ell_idx, ell2, r_idx, s_idx, s_mat = precompute_triads(wavevecs)

    # Bounds: θ ∈ [0, π/2], φ ∈ [0, 2π], ψ ∈ [0, 2π], log_a ∈ [-8, 8]
    bounds = []
    for _ in range(N):
        bounds += [(0.0, np.pi / 2), (0.0, 2 * np.pi),
                   (0.0, 2 * np.pi), (-8.0, 8.0)]

    n_workers = n_workers or os.cpu_count()
    n_workers = min(n_workers, n_starts)
    chunk  = n_starts // n_workers
    rem    = n_starts % n_workers
    counts = [chunk + (1 if i < rem else 0) for i in range(n_workers)]
    seeds  = list(range(n_workers))

    pack = (N, e1s, e2s, k2s, ell_idx, ell2, r_idx, s_idx, s_mat, bounds)

    best_val    = -np.inf
    best_params = None

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_worker, seeds[i], counts[i], *pack): i
                for i in range(n_workers)}
        for fut in as_completed(futs):
            v, p = fut.result()
            if v > best_val:
                best_val    = v
                best_params = p

    # Final polish
    args_obj = (N, e1s, e2s, k2s, ell_idx, ell2, r_idx, s_idx, s_mat)
    res = minimize(neg_ratio_and_grad, best_params, args=args_obj, method='L-BFGS-B',
                   jac=True, bounds=bounds,
                   options={'ftol': 1e-15, 'gtol': 1e-12, 'maxiter': 2000})
    best_val = float(-res.fun)

    all_shells = sorted(set(sum(ki * ki for ki in k) for k in wavevecs))
    if verbose:
        status = '✓' if best_val < 1.0 else f'✗ EXCEEDS 1.0 (+{best_val - 1.0:.4f})'
        print(f'max_shell²={max_shell2:>3}  shells={all_shells}  N={N}  '
              f'starts={n_starts}  →  max B/(K_eff·X²) = {best_val:.6f}  {status}')
    return best_val


# ─── shell scan ───────────────────────────────────────────────────────────────

def scan_shells(min_shell2=2, max_shell2=25, n_starts=300, n_workers=None):
    results = {}
    for s2 in range(min_shell2, max_shell2 + 1):
        wv = get_wavevectors(s2)
        if len(wv) < 3:          # need at least one triad
            continue
        _, ell_idx, *_ = precompute_triads(wv)
        if len(ell_idx) == 0:
            continue
        val = find_max_ratio(max_shell2=s2, n_starts=n_starts,
                             verbose=True, n_workers=n_workers)
        results[s2] = val

    print('\n' + '═' * 55)
    print('  SCAN SUMMARY  max B/(K_eff·X²)  vs shell cap')
    print('─' * 55)
    print(f'  {"shell²":>7}   {"max B/(K_eff·X²)":>18}   status')
    print('  ' + '─' * 46)
    for s2, v in sorted(results.items()):
        status = '✓' if v < 1.0 else f'✗ +{v - 1.0:.4f}'
        print(f'  {s2:>7}   {v:>18.6f}   {status}')
    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--max_shell2', type=int, default=9,
                        help='Maximum |k|² to include (default 9)')
    parser.add_argument('--min_shell2', type=int, default=2,
                        help='Minimum shell² for scan (default 2)')
    parser.add_argument('--n_starts',  type=int, default=500,
                        help='Number of random restarts (default 500)')
    parser.add_argument('--n_workers', type=int, default=None,
                        help='Worker processes (default: all cores)')
    parser.add_argument('--scan', action='store_true',
                        help='Scan min_shell2..max_shell2 incrementally')
    args = parser.parse_args()

    if args.scan:
        scan_shells(args.min_shell2, args.max_shell2,
                    args.n_starts, args.n_workers)
    else:
        find_max_ratio(args.max_shell2, args.n_starts,
                       verbose=True, n_workers=args.n_workers)
