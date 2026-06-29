"""
Find the maximizer of B/(X^2*D) over divergence-free fields,
then decompose B by output frequency shell at the maximizer.

Performance tiers (auto-selected):
  GPU  + torch: analytic gradients + CUDA tensor contractions,
                ThreadPoolExecutor (CUDA-safe, no fork).
  CPU  + torch: analytic gradients via autograd,
                multiprocessing.Pool (each worker builds torch tensors).
  CPU  + numpy: original finite-difference path (fallback if torch absent).

Analytic gradients reduce per-gradient-step cost from O(4N) function
evaluations (finite differences) to O(1) forward+backward — a ~2800x
speedup for S²=49 (N≈700 modes).
"""
import collections, glob, io, os, sys, time
import numpy as np
from scipy.optimize import minimize
import multiprocessing as mp
import concurrent.futures

# Optional torch for gradient + GPU support
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def get_modes(S2_max, S2_min=2):
    modes = []
    # Each component satisfies n_i^2 <= k^2 <= S2_max, so |n_i| <= floor(sqrt(S2_max)).
    # Use a dynamic range so no modes are silently dropped for large S2_max.
    import math
    n_max = math.isqrt(S2_max)
    for n1 in range(-n_max, n_max + 1):
        for n2 in range(-n_max, n_max + 1):
            for n3 in range(-n_max, n_max + 1):
                k2 = n1*n1 + n2*n2 + n3*n3
                if S2_min <= k2 <= S2_max:
                    modes.append((n1, n2, n3))
    return modes


def pos_half(modes):
    return [k for k in modes if (k[0] > 0 or
                                  (k[0] == 0 and k[1] > 0) or
                                  (k[0] == 0 and k[1] == 0 and k[2] > 0))]


def build_div_free_basis(k):
    k = np.array(k, dtype=float)
    kn = k / np.linalg.norm(k)
    v = np.array([1, 0, 0]) if abs(kn[0]) < 0.9 else np.array([0, 1, 0])
    e1 = v - np.dot(v, kn) * kn; e1 /= np.linalg.norm(e1)
    e2 = np.cross(kn, e1); e2 /= np.linalg.norm(e2)
    return e1, e2


def build_problem(S2):
    """Precompute all numpy arrays needed for fast vectorized evaluation."""
    modes = get_modes(S2)
    mode_set = set(modes)
    pos = pos_half(modes)
    N = len(pos)

    e1_arr = np.zeros((N, 3))
    e2_arr = np.zeros((N, 3))
    k2_arr = np.zeros(N)
    pos_idx = {k: i for i, k in enumerate(pos)}

    for i, k in enumerate(pos):
        e1, e2 = build_div_free_basis(k)
        e1_arr[i] = e1
        e2_arr[i] = e2
        k2_arr[i] = sum(x*x for x in k)

    def mode_ref(m):
        if m in pos_idx:
            return pos_idx[m], False
        nm = (-m[0], -m[1], -m[2])
        if nm in pos_idx:
            return pos_idx[nm], True
        return None, None

    ell_idxs, el2s = [], []
    r_idxs, r_conjs = [], []
    s_idxs, s_conjs = [], []
    s_vecs = []

    for i, ell in enumerate(pos):
        el2 = int(k2_arr[i])
        for r in modes:
            sv = (ell[0]-r[0], ell[1]-r[1], ell[2]-r[2])
            if sv in mode_set:
                ri, rc = mode_ref(r)
                si, sc = mode_ref(sv)
                if ri is not None and si is not None:
                    ell_idxs.append(i)
                    el2s.append(el2)
                    r_idxs.append(ri)
                    r_conjs.append(rc)
                    s_idxs.append(si)
                    s_conjs.append(sc)
                    s_vecs.append([float(sv[0]), float(sv[1]), float(sv[2])])

    return {
        'S2': S2, 'N': N, 'pos': pos,
        'k2_arr': k2_arr,
        'k4_arr': k2_arr**2,
        'e1_arr': e1_arr,
        'e2_arr': e2_arr,
        'ell_idxs': np.array(ell_idxs, dtype=np.int32),
        'el2s':     np.array(el2s,     dtype=np.float64),
        'r_idxs':   np.array(r_idxs,   dtype=np.int32),
        'r_conjs':  np.array(r_conjs,  dtype=bool),
        's_idxs':   np.array(s_idxs,   dtype=np.int32),
        's_conjs':  np.array(s_conjs,  dtype=bool),
        's_vecs':   np.array(s_vecs,   dtype=np.float64),
        'n_triads': len(ell_idxs),
    }


def make_thm158_warmstart(prob):
    """Warm start from the Theorem 15.8 exact maximizer of the minimal triad.
    û(1,1,0)=(a,-a,b), û(1,-1,0)=(c,c,d), û(2,0,0)=(0,0,f)
    at |d/a|=|b/c|=√2, f=-iφ, φ=sqrt((√17-1)/2).
    """
    N = prob['N']
    coeffs = np.zeros(4 * N)
    basis_idx = {k: i for i, k in enumerate(prob['pos'])}

    phi = np.sqrt((np.sqrt(17) - 1) / 2)   # ≈ 1.2496

    targets = {
        (1,  1, 0): np.array([1.0, -1.0,  np.sqrt(2)],        dtype=complex),
        (1, -1, 0): np.array([1.0,  1.0,  np.sqrt(2)],        dtype=complex),
        (2,  0, 0): np.array([0.0,  0.0, +1j * phi],          dtype=complex),
    }
    for k, u_target in targets.items():
        if k not in basis_idx:
            continue
        i = basis_idx[k]
        e1 = prob['e1_arr'][i]
        e2 = prob['e2_arr'][i]
        c1 = np.dot(e1, u_target)   # complex
        c2 = np.dot(e2, u_target)   # complex
        coeffs[i*4:i*4+4] = [c1.real, c1.imag, c2.real, c2.imag]
    return coeffs


def _get_u(coeffs, prob):
    """Compute div-free velocity field: returns (N, 3) complex array."""
    N = prob['N']
    c = coeffs.reshape(N, 4)
    c_e1 = (c[:, 0] + 1j * c[:, 1])[:, None]
    c_e2 = (c[:, 2] + 1j * c[:, 3])[:, None]
    return c_e1 * prob['e1_arr'] + c_e2 * prob['e2_arr']


def _eval_B_X2_D2(coeffs, prob):
    """Vectorized evaluation of B, X2, D2."""
    u = _get_u(coeffs, prob)

    # X2, D2: factor 2 accounts for both +k and -k
    amp2 = 2.0 * np.sum(np.abs(u)**2, axis=1)
    X2 = float(np.dot(prob['k2_arr'], amp2))
    D2 = float(np.dot(prob['k4_arr'], amp2))
    if X2 < 1e-14 or D2 < 1e-14:
        return 0.0, 0.0, 0.0

    # Gather u vectors for each triad, applying conjugate for negative modes
    u_ell = u[prob['ell_idxs']]
    u_r   = u[prob['r_idxs']]
    u_r   = np.where(prob['r_conjs'][:, None], u_r.conj(), u_r)
    u_s   = u[prob['s_idxs']]
    u_s   = np.where(prob['s_conjs'][:, None], u_s.conj(), u_s)

    s_dot_ur   = (prob['s_vecs'] * u_r).sum(axis=1)     # (T,) complex
    uell_dot_us = (u_ell.conj() * u_s).sum(axis=1)      # (T,) complex
    # Factor 2: B_{-ell} = B_{+ell} for fields with reality condition,
    # so the full B = 2 * (sum over positive ell only).
    B = float(2.0 * np.sum(-prob['el2s'] * np.imag(s_dot_ur * uell_dot_us)))
    return B, X2, D2


# ---------------------------------------------------------------------------
# Torch-accelerated path (analytic gradients + optional GPU)
# ---------------------------------------------------------------------------

# Triad arrays are kept on CPU and streamed to the device in chunks of this
# many triads to avoid exhausting VRAM.  Three full arrays at 81 M entries
# × 3 × complex128 = ~12 GB would exceed an 8 GB card on their own.
_TRIAD_CHUNK = 8_000_000


def _build_torch_problem(prob, device):
    """Convert numpy problem arrays to torch tensors.

    Mode-level arrays (N ≈ 9 k entries) go to `device` (GPU).
    Triad-level arrays (T ≈ 81 M entries, GB-scale) stay on CPU with
    pin_memory so they can be streamed to the device one chunk at a time.
    """
    if not _TORCH_AVAILABLE:
        return None
    td       = torch.float64
    use_cuda = (torch.device(device).type == 'cuda')

    def _r_gpu(a): return torch.tensor(a, dtype=td, device=device)
    def _r_cpu(a):
        t = torch.tensor(a, dtype=td)
        return t.pin_memory() if use_cuda else t
    def _i_cpu(a):
        t = torch.tensor(a, dtype=torch.long)
        return t.pin_memory() if use_cuda else t
    def _b_cpu(a):
        t = torch.tensor(a, dtype=torch.bool)
        return t.pin_memory() if use_cuda else t

    e1     = _r_gpu(prob['e1_arr'])
    e2     = _r_gpu(prob['e2_arr'])
    zeros3 = torch.zeros_like(e1)

    sv   = torch.tensor(prob['s_vecs'], dtype=td)
    sv_c = torch.complex(sv, torch.zeros_like(sv))
    if use_cuda:
        sv_c = sv_c.pin_memory()

    return {
        'N':        prob['N'],
        # GPU (mode-level, tiny — always resident):
        'k2':       _r_gpu(prob['k2_arr']),
        'k4':       _r_gpu(prob['k4_arr']),
        'e1_c':     torch.complex(e1, zeros3),
        'e2_c':     torch.complex(e2, zeros3),
        # CPU (triad-level, large — streamed to device in chunks):
        'el2s':     _r_cpu(prob['el2s']),
        'ell_idxs': _i_cpu(prob['ell_idxs']),
        'r_idxs':   _i_cpu(prob['r_idxs']),
        's_idxs':   _i_cpu(prob['s_idxs']),
        'r_conjs':  _b_cpu(prob['r_conjs']),
        's_conjs':  _b_cpu(prob['s_conjs']),
        's_vecs_c': sv_c,
    }


def _eval_ratio_torch(coeffs_t, tp):
    """
    Evaluation of -B/(X²·D) using PyTorch with chunked triad streaming.

    Triad arrays live on CPU and are moved to device one _TRIAD_CHUNK at a
    time so the full T×3 complex128 arrays (~12 GB at S²≈270) never coexist
    in VRAM.

    coeffs_t : 1-D real float64 tensor of shape (4N,)
    tp       : dict returned by _build_torch_problem
    Returns  : scalar tensor (negative, for minimisation)
    """
    device = coeffs_t.device
    N = tp['N']
    c = coeffs_t.view(N, 4)
    c1 = torch.view_as_complex(c[:, :2].contiguous())
    c2 = torch.view_as_complex(c[:, 2:].contiguous())
    u  = c1.unsqueeze(1) * tp['e1_c'] + c2.unsqueeze(1) * tp['e2_c']

    amp2 = 2.0 * u.abs().pow(2).sum(dim=1)
    X2   = (tp['k2'] * amp2).sum()
    D2   = (tp['k4'] * amp2).sum()
    if X2 < 1e-28 or D2 < 1e-28:
        return torch.zeros(1, dtype=torch.float64, device=device).squeeze()

    # Stream triads from CPU → device, accumulate B without keeping all of
    # u_ell / u_r / u_s resident in VRAM simultaneously.
    u_d = u.detach()
    T   = tp['ell_idxs'].shape[0]
    B   = torch.zeros(1, dtype=torch.float64, device=device)
    for start in range(0, T, _TRIAD_CHUNK):
        end     = min(start + _TRIAD_CHUNK, T)
        ell_i   = tp['ell_idxs'][start:end].to(device, non_blocking=True)
        r_i     = tp['r_idxs'][start:end].to(device, non_blocking=True)
        s_i     = tp['s_idxs'][start:end].to(device, non_blocking=True)
        rc      = tp['r_conjs'][start:end].to(device, non_blocking=True)
        sc      = tp['s_conjs'][start:end].to(device, non_blocking=True)
        el2     = tp['el2s'][start:end].to(device, non_blocking=True)
        svec    = tp['s_vecs_c'][start:end].to(device, non_blocking=True)
        u_ell_c = u_d[ell_i]
        u_r_c   = u_d[r_i]
        u_s_c   = u_d[s_i]
        u_r_c   = torch.where(rc.unsqueeze(1), u_r_c.conj(), u_r_c)
        u_s_c   = torch.where(sc.unsqueeze(1), u_s_c.conj(), u_s_c)
        sdur    = (svec * u_r_c).sum(dim=1)
        uellus  = (u_ell_c.conj() * u_s_c).sum(dim=1)
        B      += 2.0 * (-el2 * (sdur * uellus).imag).sum()
        del ell_i, r_i, s_i, rc, sc, el2, svec, u_ell_c, u_r_c, u_s_c, sdur, uellus

    return -B.squeeze() / (X2 * D2.sqrt())


def _eval_neg_ratio_and_grad_torch(coeffs_np, tp, device):
    """Return value and gradient for -B/(X^2 D) with streamed B gradients.

    The value path can stream detached triad chunks, but L-BFGS needs the true
    B-gradient.  We accumulate dB/du on a leaf copy of the velocity field chunk
    by chunk, then chain that gradient through the small u(coeffs) graph.
    """
    coeffs_t = torch.tensor(coeffs_np, dtype=torch.float64, device=device, requires_grad=True)
    N = tp['N']
    c = coeffs_t.view(N, 4)
    c1 = torch.view_as_complex(c[:, :2].contiguous())
    c2 = torch.view_as_complex(c[:, 2:].contiguous())
    u = c1.unsqueeze(1) * tp['e1_c'] + c2.unsqueeze(1) * tp['e2_c']

    amp2 = 2.0 * u.abs().pow(2).sum(dim=1)
    X2 = (tp['k2'] * amp2).sum()
    D2 = (tp['k4'] * amp2).sum()
    if X2 < 1e-28 or D2 < 1e-28:
        return 0.0, np.zeros_like(coeffs_np)

    D = D2.sqrt()
    u_leaf = u.detach().clone().requires_grad_(True)
    T = tp['ell_idxs'].shape[0]
    B_accum = 0.0
    for start in range(0, T, _TRIAD_CHUNK):
        end = min(start + _TRIAD_CHUNK, T)
        ell_i = tp['ell_idxs'][start:end].to(device, non_blocking=True)
        r_i = tp['r_idxs'][start:end].to(device, non_blocking=True)
        s_i = tp['s_idxs'][start:end].to(device, non_blocking=True)
        rc = tp['r_conjs'][start:end].to(device, non_blocking=True)
        sc = tp['s_conjs'][start:end].to(device, non_blocking=True)
        el2 = tp['el2s'][start:end].to(device, non_blocking=True)
        svec = tp['s_vecs_c'][start:end].to(device, non_blocking=True)
        u_ell_c = u_leaf[ell_i]
        u_r_c = u_leaf[r_i]
        u_s_c = u_leaf[s_i]
        u_r_c = torch.where(rc.unsqueeze(1), u_r_c.conj(), u_r_c)
        u_s_c = torch.where(sc.unsqueeze(1), u_s_c.conj(), u_s_c)
        sdur = (svec * u_r_c).sum(dim=1)
        uellus = (u_ell_c.conj() * u_s_c).sum(dim=1)
        B_chunk = 2.0 * (-el2 * (sdur * uellus).imag).sum()
        B_accum += B_chunk.item()
        B_chunk.backward()
        del ell_i, r_i, s_i, rc, sc, el2, svec, u_ell_c, u_r_c, u_s_c, sdur, uellus, B_chunk

    u.backward(gradient=u_leaf.grad)
    dB = coeffs_t.grad.detach()
    X2_det = X2.detach()
    D2_det = D2.detach()
    D_det = D.detach()
    R = torch.tensor(B_accum, dtype=torch.float64, device=device) / (X2_det * D_det)
    k2_4 = tp['k2'].repeat_interleave(4)
    k4_4 = tp['k4'].repeat_interleave(4)
    coeffs_det = coeffs_t.detach()
    grad_R = dB / (X2_det * D_det) - R * (
        4.0 * k2_4 * coeffs_det / X2_det
        + 2.0 * k4_4 * coeffs_det / D2_det
    )
    return float(-R.cpu()), (-grad_R).cpu().numpy().copy()


# --- Worker globals (avoids re-pickling prob for every task) ---
_PROB        = None   # numpy problem (original path)
_TORCH_PROB  = None   # torch tensors  (gradient path)
_TORCH_DEV   = None   # torch.device


def _worker_init(prob):
    global _PROB
    _PROB = prob


def _worker_init_torch(prob_raw, device_str):
    """Initialiser for process-pool workers when using the torch CPU path."""
    global _PROB, _TORCH_PROB, _TORCH_DEV
    _PROB       = prob_raw
    _TORCH_DEV  = torch.device(device_str)
    _TORCH_PROB = _build_torch_problem(prob_raw, _TORCH_DEV)


def _neg_ratio_global(coeffs):
    B, X2, D2 = _eval_B_X2_D2(coeffs, _PROB)
    if X2 < 1e-14 or D2 < 1e-14:
        return 0.0
    return -B / (X2 * np.sqrt(D2))


def _run_start(z0):
    """Numpy finite-difference L-BFGS-B (fallback when torch unavailable)."""
    try:
        res = minimize(_neg_ratio_global, z0, method='L-BFGS-B',
                       options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-9})
        return float(-res.fun), res.x
    except Exception:
        return float(-np.inf), z0


def _run_start_torch(z0):
    """Torch analytic-gradient L-BFGS-B (used by both CPU and GPU paths)."""
    tp     = _TORCH_PROB
    device = _TORCH_DEV
    try:
        def f_and_g(c_np):
            return _eval_neg_ratio_and_grad_torch(c_np, tp, device)

        res = minimize(f_and_g, z0, method='L-BFGS-B', jac=True,
                       options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-9})
        return float(-res.fun), res.x
    except Exception:
        return float(-np.inf), z0


def _newton_polish(init_coeffs, tp, device, n_sweeps=20, damp=0.9, tol=1e-11):
    """Coordinate Newton polish for R = B / (X² D).

    At a critical point the per-mode optimality condition reduces to a
    closed-form fixed-point equation:

        p_k* = (∂B/∂p_k) / (R · (4 k² D + 2 k⁴ X²/D))

    where p_k = [Re(c1), Im(c1), Re(c2), Im(c2)] are the 4 real parameters
    for mode k.  Applied simultaneously over all N modes, one sweep costs
    exactly one autograd backward pass (same as one L-BFGS step), but near
    the optimum converges quadratically — empirically 10–20 sweeps reach 1e-14
    vs ~3000 L-BFGS steps.  Expected speedup: 50–300×.

    Parameters
    ----------
    init_coeffs : (4N,) numpy array — starting point (L-BFGS output)
    tp          : torch problem dict from _build_torch_problem
    device      : torch.device
    n_sweeps    : max coordinate-Newton sweeps
    damp        : step damping ∈ (0, 1] — 1.0 = full Newton
    tol         : stop when |ΔR|/R < tol between sweeps
    """
    k2 = tp['k2']   # (N,)
    k4 = tp['k4']   # (N,)
    N  = tp['N']

    coeffs = torch.tensor(init_coeffs, dtype=torch.float64, device=device)

    with torch.no_grad():
        R0 = float(-_eval_ratio_torch(coeffs, tp).item())

    best_val  = R0
    best_np   = init_coeffs.copy()
    prev_val  = R0
    no_improve_count = 0

    T_triads = tp['ell_idxs'].shape[0]
    for sweep in range(n_sweeps):
        # ── Small graph: c_rg → u (N×3 complex, <1 MB on GPU) ─────────────
        c_rg = coeffs.detach().clone().requires_grad_(True)
        cv   = c_rg.view(N, 4)
        c1   = torch.view_as_complex(cv[:, :2].contiguous())
        c2   = torch.view_as_complex(cv[:, 2:].contiguous())
        u    = c1.unsqueeze(1) * tp['e1_c'] + c2.unsqueeze(1) * tp['e2_c']

        amp2 = 2.0 * u.abs().pow(2).sum(dim=1)
        X2   = (k2 * amp2).sum()
        D2   = (k4 * amp2).sum()
        if X2 < 1e-28 or D2 < 1e-28:
            break
        D = D2.sqrt()

        # ── Chunked B + gradient accumulation ──────────────────────────────
        # Triad arrays live on CPU; stream one _TRIAD_CHUNK at a time to GPU.
        # u_leaf mirrors u.detach() but is a leaf: its .grad accumulates
        # dB/du* across all chunks, replacing one monolithic B.backward().
        # Afterwards, u.backward(u_leaf.grad) chain-rules through the tiny
        # u = f(c_rg) graph to produce dB/dc_rg without ever holding all
        # triad expanded tensors in VRAM simultaneously.
        u_leaf  = u.detach().clone().requires_grad_(True)
        B_accum = 0.0
        for start in range(0, T_triads, _TRIAD_CHUNK):
            end     = min(start + _TRIAD_CHUNK, T_triads)
            ell_i   = tp['ell_idxs'][start:end].to(device, non_blocking=True)
            r_i     = tp['r_idxs'][start:end].to(device, non_blocking=True)
            s_i     = tp['s_idxs'][start:end].to(device, non_blocking=True)
            rc      = tp['r_conjs'][start:end].to(device, non_blocking=True)
            sc      = tp['s_conjs'][start:end].to(device, non_blocking=True)
            el2     = tp['el2s'][start:end].to(device, non_blocking=True)
            svec    = tp['s_vecs_c'][start:end].to(device, non_blocking=True)
            u_ell_c = u_leaf[ell_i]
            u_r_c   = u_leaf[r_i]
            u_s_c   = u_leaf[s_i]
            u_r_c   = torch.where(rc.unsqueeze(1), u_r_c.conj(), u_r_c)
            u_s_c   = torch.where(sc.unsqueeze(1), u_s_c.conj(), u_s_c)
            sdur    = (svec * u_r_c).sum(dim=1)
            uellus  = (u_ell_c.conj() * u_s_c).sum(dim=1)
            B_chunk = 2.0 * (-el2 * (sdur * uellus).imag).sum()
            B_accum += B_chunk.item()
            B_chunk.backward()   # accumulates u_leaf.grad; frees chunk graph
            del ell_i, r_i, s_i, rc, sc, el2, svec
            del u_ell_c, u_r_c, u_s_c, sdur, uellus, B_chunk

        if B_accum < 1e-14:
            break

        X2_val  = X2.detach()
        D_val   = D.detach()
        R_t_val = torch.tensor(B_accum, dtype=torch.float64,
                               device=device) / (X2_val * D_val)

        # ── Chain rule: dB/dc_rg via backward through tiny u = f(c_rg) ─────
        u.backward(gradient=u_leaf.grad)
        dB = c_rg.grad.detach()   # (4N,)
        del u, c1, c2, cv, amp2, X2, D2, D, u_leaf, c_rg

        with torch.no_grad():
            # Λ_k = R × (4 k² D + 2 k⁴ X²/D) — positive scalar per mode
            Lam   = R_t_val * (4.0 * k2 * D_val + 2.0 * k4 * X2_val / D_val)  # (N,)
            Lam4N = Lam.repeat_interleave(4)                    # (4N,)

            # Closed-form Newton step: p_k* = dB_k / Λ_k
            p_new = dB / Lam4N

            # Damped candidate
            c_new   = damp * p_new + (1.0 - damp) * coeffs
            new_val = float(-_eval_ratio_torch(c_new, tp).item())

            if new_val <= best_val:
                # Damping hurt — try full Newton step
                full_val = float(-_eval_ratio_torch(p_new, tp).item())
                if full_val > new_val:
                    new_val, c_new = full_val, p_new

            if new_val > best_val:
                best_val = new_val
                best_np  = c_new.cpu().numpy()
                coeffs   = c_new
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= 3:
                    break

        # Convergence: fractional improvement vanished
        if best_val > 1e-14 and (best_val - prev_val) < tol * prev_val:
            break
        prev_val = best_val

    return best_val, best_np


def _save_coeffs(S2, best_z, best_val, prob):
    """Save optimal coefficients to disk, only if strictly better than existing.

    Guards against accidentally overwriting a better solution with a weaker one.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    coeff_dir = os.path.join(script_dir, 'results', 'opt_coeffs')
    os.makedirs(coeff_dir, exist_ok=True)
    path = os.path.join(coeff_dir, f'opt_coeffs_s{S2}.npy')
    if os.path.exists(path):
        try:
            existing_z = np.load(path)
            ex_B, ex_X2, ex_D2 = _eval_B_X2_D2(existing_z, prob)
            existing_val = (ex_B / (ex_X2 * np.sqrt(ex_D2))
                            if ex_X2 > 1e-14 and ex_D2 > 1e-14 else 0.0)
            if existing_val >= best_val:
                print(f"  Existing coeffs ({existing_val:.17g}) >= new ({best_val:.17g}); "
                      f"skipping overwrite — existing file preserved.", flush=True)
                return path
            print(f"  New value {best_val:.17g} > existing {existing_val:.17g}; overwriting.",
                  flush=True)
        except Exception as e:
            print(f"  Could not evaluate existing coeffs ({e}); overwriting.", flush=True)
    np.save(path, best_z)
    return path


def _is_legendre(n):
    """True iff n cannot be represented as a sum of 3 integer squares
    (three-squares theorem: n = 4^a*(8b+7) for some a,b >= 0).
    Equivalently: no lattice points at radius sqrt(n), so build_problem(n)
    has the same mode set as build_problem(n-1).
    """
    while n % 4 == 0:
        n //= 4
    return n % 8 == 7


def _load_best_warmstart(S2, prob):
    """Load optimal coefficients from the largest completed S2_prev < S2 and embed them.
    Modes common to both problems share e1/e2 basis vectors (deterministic), so
    coefficients can be copied directly; new modes are initialised to zero.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    found = []
    coeff_dirs = [os.path.join(script_dir, 'results', 'opt_coeffs'), script_dir]
    for coeff_dir in coeff_dirs:
        for p in glob.glob(os.path.join(coeff_dir, 'opt_coeffs_s*.npy')):
            try:
                s2_val = int(os.path.basename(p)[len('opt_coeffs_s'):-len('.npy')])
                if s2_val < S2:
                    found.append((s2_val, p))
            except ValueError:
                pass
    if not found:
        return None
    best_s2, best_path = max(found, key=lambda x: x[0])
    prev_coeffs = np.load(best_path)
    prev_prob = build_problem(best_s2)
    N_curr = prob['N']
    embedded = np.zeros(4 * N_curr)
    curr_idx = {k: i for i, k in enumerate(prob['pos'])}
    for i, k in enumerate(prev_prob['pos']):
        if k in curr_idx:
            j = curr_idx[k]
            embedded[j*4:j*4+4] = prev_coeffs[i*4:i*4+4]
    return embedded, best_s2


def analyze(S2, n_starts=300, seed=42, n_workers=None, device=None):
    t0 = time.time()
    print(f"\n{'='*60}", flush=True)
    print(f"S^2 = {S2}", flush=True)

    # ---- device / path selection ----------------------------------------
    if _TORCH_AVAILABLE:
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dev = torch.device(device)
        use_torch = True
        use_gpu   = (dev.type == 'cuda')
    else:
        use_torch = False
        use_gpu   = False
        device    = 'cpu'

    print(f"  Backend: {'torch/' + device if use_torch else 'numpy/cpu'}", flush=True)

    prob = build_problem(S2)
    N = prob['N']
    print(f"  {prob['n_triads']} triads, {N} positive modes", flush=True)
    print(f"  Setup: {time.time()-t0:.1f}s", flush=True)

    # ---- initialise worker globals in main process ----------------------
    if use_torch:
        global _TORCH_PROB, _TORCH_DEV
        # Free previous shell's GPU tensors before allocating the new ones.
        # Without this, Python may hold the old _TORCH_PROB alive until after
        # the new one is built, causing OOM on large shells (S²≳260).
        if _TORCH_PROB is not None:
            del _TORCH_PROB
            _TORCH_PROB = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        _TORCH_PROB = _build_torch_problem(prob, dev)
        _TORCH_DEV  = dev
        _run_one = _run_start_torch
    else:
        _worker_init(prob)   # numpy path: set _PROB in main process
        _run_one = _run_start

    if n_workers is None:
        n_workers = min(mp.cpu_count(), n_starts)

    rng = np.random.default_rng(seed)
    t1  = time.time()

    # ── Legendre fast path ─────────────────────────────────────────────────
    # S² = 4^a*(8b+7) has no lattice points at radius sqrt(S²), so
    # build_problem(S²) is identical to build_problem(S²-1).  No optimisation
    # is needed: just copy the previous shell's coefficients.
    if _is_legendre(S2):
        embedded_result = _load_best_warmstart(S2, prob)
        if embedded_result is not None:
            embedded_z, prev_s2 = embedded_result
            B, X2, D2 = _eval_B_X2_D2(embedded_z, prob)
            best_val = B / (X2 * D2 ** 0.5) if X2 > 1e-14 and D2 > 1e-14 else 0.0
            _save_coeffs(S2, embedded_z, best_val, prob)
            print(f"  [Legendre: no modes at |k|²={S2} — plateau from S²={prev_s2}]", flush=True)
            print(f"\n  MAX B/(X^2*D) = {best_val:.17g}", flush=True)
            print(f"  Total: {time.time()-t0:.1f}s", flush=True)
            return
        print(f"  [Legendre shell but no prior coeffs found — running optimization]", flush=True)

    # ── Fast sequential mode (n_starts == 1) ──────────────────────────────
    # When doing a dense chain scan, warm-start from the previous shell and
    # run L-BFGS with a tight iteration budget.  From a near-optimal start
    # the optimizer converges in ~50-100 iterations vs ~3000 for a cold
    # start: ~1-2 min/shell instead of ~25 min/shell for S²≈200.
    embedded_result = _load_best_warmstart(S2, prob)
    if n_starts == 1 and embedded_result is not None and use_torch:
        embedded_z, embedded_s2 = embedded_result
        print(f"  Warm-start: S²={embedded_s2}; running Newton sweeps (n=40)...",
              flush=True)
        emb_val, emb_opt_z = _newton_polish(
            embedded_z, _TORCH_PROB, dev,
            n_sweeps=40, damp=0.95, tol=1e-10,
        )
        print(f"  Embedded S²={embedded_s2} (Newton-40): {emb_val:.17g}", flush=True)
        best_val, best_z = emb_val, emb_opt_z
        print(f"  Optimization: {time.time()-t1:.1f}s", flush=True)
    elif n_starts == 1 and embedded_result is not None:
        # numpy fallback: keep L-BFGS
        embedded_z, embedded_s2 = embedded_result
        emb_val, emb_opt_z = _run_one(embedded_z)
        print(f"  Embedded S²={embedded_s2} start: {emb_val:.17g}", flush=True)
        best_val, best_z = emb_val, emb_opt_z
        print(f"  Optimization: {time.time()-t1:.1f}s", flush=True)
    else:
        # ── Full multi-start path ──────────────────────────────────────────
        warm = make_thm158_warmstart(prob)
        warm_val, warm_opt_z = _run_one(warm)
        print(f"  Thm 15.8 warm-start value: {warm_val:.17g}", flush=True)
        starts   = [warm, warm_opt_z]
        n_random = n_starts - 2

        if embedded_result is not None:
            embedded_z, embedded_s2 = embedded_result
            emb_val, emb_opt_z = _run_one(embedded_z)
            print(f"  Embedded S²={embedded_s2} warm-start: {emb_val:.17g}", flush=True)
            starts.extend([embedded_z, emb_opt_z])
            n_random -= 2

        starts += [rng.standard_normal(4 * N) for _ in range(max(0, n_random))]

        n_actual = len(starts)
        print(f"  Running {n_actual} starts on {n_workers} workers "
              f"({'threads/GPU' if use_gpu else 'processes/CPU'})...", flush=True)

        if use_gpu:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
                futures = [ex.submit(_run_start_torch, z) for z in starts]
                results = []
                for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                    result = future.result()
                    results.append(result)
                    if i % 10 == 0 or i == n_actual:
                        best_so_far = max(results, key=lambda x: x[0])[0]
                        print(f"    [{i}/{n_actual}] best so far: {best_so_far:.17g}  (+{time.time()-t1:.0f}s)", flush=True)
        elif use_torch:
            ctx = mp.get_context('spawn')
            with ctx.Pool(processes=n_workers,
                          initializer=_worker_init_torch,
                          initargs=(prob, device)) as pool:
                results = []
                for i, result in enumerate(pool.imap_unordered(_run_start_torch, starts,
                                           chunksize=max(1, n_actual // (4 * n_workers))), 1):
                    results.append(result)
                    if i % 10 == 0 or i == n_actual:
                        best_so_far = max(results, key=lambda x: x[0])[0]
                        print(f"    [{i}/{n_actual}] best so far: {best_so_far:.17g}  (+{time.time()-t1:.0f}s)", flush=True)
        else:
            with mp.Pool(processes=n_workers,
                         initializer=_worker_init,
                         initargs=(prob,)) as pool:
                results = []
                for i, result in enumerate(pool.imap_unordered(_run_start, starts,
                                           chunksize=max(1, n_actual // (4 * n_workers))), 1):
                    results.append(result)
                    if i % 10 == 0 or i == n_actual:
                        best_so_far = max(results, key=lambda x: x[0])[0]
                        print(f"    [{i}/{n_actual}] best so far: {best_so_far:.17g}  (+{time.time()-t1:.0f}s)", flush=True)

        best_val, best_z = max(results, key=lambda x: x[0])
        print(f"  Optimization: {time.time()-t1:.1f}s", flush=True)

    # ── Newton polish (multi-start path only) ────────────────────────────────
    # n_starts==1 warm-start path already ran Newton-40 sweeps above.
    # For multi-start, polish the best L-BFGS result with Newton sweeps to higher precision.
    if use_torch and best_val > 1e-14 and not (n_starts == 1 and embedded_result is not None):
        t_np = time.time()
        np_val, np_z = _newton_polish(best_z, _TORCH_PROB, dev)
        delta = np_val - best_val
        if delta > 0:
            print(f"  Newton polish: {best_val:.17g} → {np_val:.17g} "
                  f"(Δ={delta:+.4e}, {time.time()-t_np:.1f}s)", flush=True)
            best_val, best_z = np_val, np_z
        else:
            print(f"  Newton polish: no improvement ({time.time()-t_np:.1f}s)", flush=True)

    coeffs_path = _save_coeffs(S2, best_z, best_val, prob)
    print(f"  Saved optimal coeffs: {os.path.basename(coeffs_path)}", flush=True)
    print(f"\n  MAX B/(X^2*D) = {best_val:.17g}", flush=True)

    # Full shell decomposition at the maximizer (main process already init'd)
    u = _get_u(best_z, prob)
    amp2 = 2.0 * np.sum(np.abs(u)**2, axis=1)
    X2 = float(np.dot(prob['k2_arr'], amp2))
    D2 = float(np.dot(prob['k4_arr'], amp2))
    D  = np.sqrt(D2)

    u_ell = u[prob['ell_idxs']]
    u_r   = u[prob['r_idxs']]
    u_r   = np.where(prob['r_conjs'][:, None], u_r.conj(), u_r)
    u_s   = u[prob['s_idxs']]
    u_s   = np.where(prob['s_conjs'][:, None], u_s.conj(), u_s)
    s_dot_ur    = (prob['s_vecs'] * u_r).sum(axis=1)
    uell_dot_us = (u_ell.conj() * u_s).sum(axis=1)
    B_contribs  = 2.0 * (-prob['el2s'] * np.imag(s_dot_ur * uell_dot_us))
    B = float(np.sum(B_contribs))

    B_sh = collections.defaultdict(float)
    for el2, c in zip(prob['el2s'], B_contribs):
        B_sh[int(el2)] += float(c)

    print(f"  Verification: B/(X^2*D) = {B/(X2*D):.17g}", flush=True)
    print(f"\n  Per-shell breakdown at maximizer:", flush=True)
    print(f"  {'|ell|^2':>10}  {'B_sh':>14}  {'B_sh/(X^2*D)':>16}  {'fraction':>10}", flush=True)
    for sh in sorted(B_sh):
        frac = B_sh[sh] / B if abs(B) > 1e-14 else float('nan')
        print(f"  {sh:>10}  {B_sh[sh]:>14.6f}  {B_sh[sh]/(X2*D):>16.8f}  {frac:>10.4f}", flush=True)
    print(f"  {'TOTAL':>10}  {B:>14.6f}  {B/(X2*D):>16.17g}  {'1.0000':>10}", flush=True)
    print(f"  Total time: {time.time()-t0:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

class _TeeIO:
    """Write to both the original stdout and an internal buffer."""
    def __init__(self, target):
        self._target = target
        self._buf    = io.StringIO()

    def write(self, s):
        self._target.write(s)
        self._buf.write(s)

    def flush(self):
        self._target.flush()

    def getvalue(self):
        return self._buf.getvalue()

    def __getattr__(self, name):
        return getattr(self._target, name)


def _save_run_log(run_start_ts, run_start_hms, S2, n_starts, log_text, args_str):
    """Write captured output to scripts/results/YYYY-MM-DD/ using gap3 naming convention."""
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    today       = run_start_ts[:10]
    results_dir = os.path.join(script_dir, 'results', today)
    os.makedirs(results_dir, exist_ok=True)
    fname = f'shell_decomp_s{S2}_{n_starts}starts_{run_start_hms}.txt'
    fpath = os.path.join(results_dir, fname)
    header = (
        f'Run started : {run_start_ts}\n'
        f'Script      : shell_decomp\n'
        f'Log file    : {fpath}\n'
        f'Arguments   : {args_str}\n\n'
    )
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(header)
        f.write(log_text)
    # Print to real stdout (tee already restored by caller)
    print(f'  Log saved : results/{today}/{fname}', flush=True)
    return fpath


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Multi-shell trilinear supremum optimiser')
    parser.add_argument('s2',      type=int, nargs='?',
                        help='S² value to run (omit to run all defaults)')
    parser.add_argument('starts',  type=int, nargs='?',
                        help='Number of random starts (default per-S² or 400)')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default=None,
                        help='Force device (default: cuda if available, else cpu)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: cpu_count)')
    args = parser.parse_args()

    default_runs = [
        (4,   300),
        (9,   200),
        (16,  100),
        (25,  150),
        (36,  150),
        (49,  400),
        (64,  400),
    ]

    # Build the arguments string for the log header
    _args_str = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else '(defaults)'

    def _run_logged(S2, n_starts):
        """Run analyze() with stdout tee'd to a log file."""
        run_start_ts  = time.strftime('%Y-%m-%d %H:%M:%S')
        run_start_hms = time.strftime('%H%M%S')
        tee = _TeeIO(sys.stdout)
        sys.stdout = tee
        try:
            analyze(S2=S2, n_starts=n_starts,
                    device=args.device, n_workers=args.workers)
        finally:
            sys.stdout = tee._target
            _save_run_log(run_start_ts, run_start_hms, S2, n_starts,
                          tee.getvalue(), _args_str)

    if args.s2 is not None:
        n = args.starts if args.starts is not None else 400
        _run_logged(S2=args.s2, n_starts=n)
    else:
        for S2, n in default_runs:
            _run_logged(S2=S2, n_starts=args.starts if args.starts is not None else n)
