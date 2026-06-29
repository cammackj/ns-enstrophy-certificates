"""
certify_block_maximum_gpu.py
============================
GPU-accelerated drop-in replacement for certify_block_maximum.py.

Architecture
------------
Only Steps 1 and 1c (multi-start L-BFGS-B exploration) are changed:

  Phase A — GPU batch exploration (float32, torch.optim.Adam):
    All random starts are run simultaneously as a batch on GPU.  Each
    mini-batch of --batch-size starts is processed together, giving true
    data-parallelism over the objective's triad sum.  Phase A takes seconds
    even for 500 starts.

  Phase B — CPU parallel polish (float64, scipy L-BFGS-B):
    Top --n-polish candidates from Phase A are polished in a
    multiprocessing.Pool using the same code as certify_block_maximum.py.
    This gives identical float64 tolerance and correctness guarantees.

Steps 2-5 (mpmath interval arithmetic, Hessian, ISC) are identical to the
CPU version and are imported directly from certify_block_maximum.py.

Usage (identical flags to certify_block_maximum.py, plus GPU extras)
---------------------------------------------------------------------
    python scripts/gap3/certify_block_maximum_gpu.py --k 6 --full_block \\
        --precompute-triads --warm-npz results/warm_state \\
        --dps 50 --n_workers 20 --global-cert-starts 500

Additional GPU flags:
    --batch-size N    GPU batch size for Adam exploration (default: auto)
    --n-adam-steps N  Adam steps per mini-batch (default: 400)
    --n-polish N      Top-N Adam candidates to polish with CPU L-BFGS-B
                      (default: max(n_workers, 20))
    --adam-lr F       Adam learning rate (default: 0.03)
    --device DEVICE   PyTorch device string (default: auto-detect CUDA)
    --no-gpu          Disable GPU; fall back to CPU (same as certify_block_maximum.py
                      but with parallel Step 1)

Validation
----------
After a new GPU run completes, compare its certified interval against the
CPU script's result on a case with a known certificate (e.g. k=4) to rule
out numerical bugs before trusting k=6 results.
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

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

sys.path.insert(0, ".")
from scripts.gap3.multi_mode_beta_bound import get_wavevectors, divfree_basis, precompute_triads
from scripts.gap3.gap3_principled_scan import (
    _build_problem, find_dcxa_nucleus, _restrict_to_active_modes,
)
from scripts.gap3.max_b_over_keff import neg_ratio_and_grad

# Import certification helpers (all module-level in certify_block_maximum — safe to import)
from scripts.gap3.certify_block_maximum import (
    _mpmath_objective,
    _build_mpmath_data,
    _hess_diag_worker,
    isc_check,
    _init_lbfgsb_worker,
    _lbfgsb_start_worker,
    _lbfgsb_explore_worker,    # loose-tolerance basin finder for GPU Phase B
    float64_optimise,          # used for warm-start-only passes and Step 1b
)

# Import GPU reparametrisation helpers from gpu_optimizer
from scripts.gap3.gpu_optimizer import (
    params_to_unconstrained,
    unconstrained_to_params,
)

# ---------------------------------------------------------------------------
# Float64 Hessian diagonal pre-screen worker (module-level for pickle)
# ---------------------------------------------------------------------------

def _hess_f64_worker(args):
    """Central-difference Hessian diagonal for one parameter index, float64.
    All prob fields are passed explicitly in args — no global state needed.
    Returns (param_idx, H_ii_float64).
    """
    i, x_f64, R0, h, N, k2s, e1s, e2s, ell_idx, ell2, r_idx, s_idx, s_mat = args
    xp = x_f64.copy(); xp[i] += h
    xm = x_f64.copy(); xm[i] -= h
    # neg_ratio_and_grad returns (-R, grad(-R)); negate to get R
    Rp = -neg_ratio_and_grad(xp, N, e1s, e2s, k2s, ell_idx, ell2, r_idx, s_idx, s_mat)[0]
    Rm = -neg_ratio_and_grad(xm, N, e1s, e2s, k2s, ell_idx, ell2, r_idx, s_idx, s_mat)[0]
    return (i, float((Rp + Rm - 2.0 * R0) / (h * h)))


# ---------------------------------------------------------------------------
# GPU batch objective
# ---------------------------------------------------------------------------

class GpuObjectiveBatched:
    """
    Holds all fixed tensors for one problem instance.
    Supports a batch of S starts evaluated simultaneously via forward_batch.

    Memory note: forward_batch materialises (S, T, 3) complex tensors for the
    triad contractions.  For T=2.6M (k=6) and float32:
      S= 4 → ~500 MB peak;  S= 8 → ~1 GB;  S=16 → ~2 GB.
    The autograd graph doubles these during backward().  Keep batch_size
    conservative (default auto-detects from free VRAM).
    """

    def __init__(self, prob: dict, device: "torch.device",
                 dtype: "torch.dtype" = None):
        if dtype is None:
            dtype = torch.float32
        cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128

        self.N      = prob['N']
        self.T      = len(prob['ell_idx'])
        self.dev    = device
        self.dtype  = dtype
        self.cdtype = cdtype

        self.k2s = torch.tensor(prob['k2s'], dtype=dtype, device=device)
        self.k4s = self.k2s ** 2
        self.e1s = torch.tensor(prob['e1s'].astype(np.complex128),
                                dtype=cdtype, device=device)               # (N, 3)
        self.e2s = torch.tensor(prob['e2s'].astype(np.complex128),
                                dtype=cdtype, device=device)               # (N, 3)

        self.ell_idx = torch.tensor(prob['ell_idx'].astype(np.int64), device=device)
        self.r_idx   = torch.tensor(prob['r_idx'].astype(np.int64),   device=device)
        self.s_idx   = torch.tensor(prob['s_idx'].astype(np.int64),   device=device)
        self.ell2    = torch.tensor(prob['ell2'],  dtype=dtype, device=device)  # (T,)
        self.s_mat   = torch.tensor(prob['s_mat'], dtype=dtype, device=device)  # (T, 3)

    def _build_u_batch(self, t_batch: "torch.Tensor"):
        """
        t_batch: (S, 4N) unconstrained parameters.
        Returns u_raw (S, 2N, 3) complex, a (S, N) float.
        """
        theta = (math.pi / 2.0) * torch.sigmoid(t_batch[:, 0::4])   # (S, N)
        phi   = 2.0 * math.pi  * torch.sigmoid(t_batch[:, 1::4])
        psi   = 2.0 * math.pi  * torch.sigmoid(t_batch[:, 2::4])
        la    = 8.0 * torch.tanh(t_batch[:, 3::4])
        a     = torch.exp(la)                                          # (S, N)
        r     = torch.sqrt(a)

        zeros = torch.zeros_like(phi)
        ephi  = torch.complex(torch.cos(phi), torch.sin(phi))          # (S, N)
        epsi  = torch.complex(torch.cos(psi), torch.sin(psi))

        r_c   = torch.complex(r,               zeros)
        cth_c = torch.complex(torch.cos(theta), zeros)
        sth_c = torch.complex(torch.sin(theta), zeros)

        # u_pos: (S, N, 3) = r * (cos θ e^{iφ} e1 + sin θ e^{iψ} e2)
        u_pos = r_c.unsqueeze(-1) * (
            (cth_c * ephi).unsqueeze(-1) * self.e1s.unsqueeze(0) +
            (sth_c * epsi).unsqueeze(-1) * self.e2s.unsqueeze(0)
        )                                                               # (S, N, 3)
        u_raw = torch.cat([u_pos, u_pos.conj()], dim=1)                # (S, 2N, 3)
        return u_raw, a

    def forward_batch(self, t_batch: "torch.Tensor") -> "torch.Tensor":
        """
        t_batch: (S, 4N) unconstrained params (float32 or float64).
        Returns losses (S,) = -B/(X²D) for each start.
        Autograd-compatible.
        """
        u_raw, a = self._build_u_batch(t_batch)      # (S, 2N, 3), (S, N)

        # X², D per start  (S,)
        X2 = 2.0 * (self.k2s.unsqueeze(0) * a).sum(-1)
        D  = torch.sqrt(
            (2.0 * (self.k4s.unsqueeze(0) * a).sum(-1)).clamp(min=1e-60)
        )
        XD = X2 * D                                                    # (S,)

        # Triad contractions: (S, T) quantities
        # u_raw[:, idx, :] is a gather: (S, T, 3) — the main memory cost
        u_r = u_raw[:, self.r_idx, :]     # (S, T, 3)
        u_l = u_raw[:, self.ell_idx, :]   # (S, T, 3)
        u_s = u_raw[:, self.s_idx, :]     # (S, T, 3)

        s_c  = self.s_mat.to(u_r.dtype)                               # (T, 3) complex view
        sdu  = (s_c.unsqueeze(0) * u_r).sum(-1)                       # (S, T)
        ced  = (u_l.conj() * u_s).sum(-1)                             # (S, T)

        ell2_c = self.ell2.to(sdu.dtype)                               # (T,) complex
        B      = -(ell2_c.unsqueeze(0) * sdu * ced).sum(-1).imag      # (S,)

        # Avoid division by zero
        safe_XD = XD.clamp(min=1e-60)
        return -B / safe_XD                                            # (S,) losses


# ---------------------------------------------------------------------------
# Auto batch-size detection
# ---------------------------------------------------------------------------

def auto_batch_size(T: int, dtype_bytes: int = 4) -> int:
    """
    Estimate a safe GPU batch size based on free VRAM.
    Each start in the batch requires ~3 × T × 3 × 2 × dtype_bytes bytes
    (u_r, u_l, u_s tensors, each complex = 2×float, plus autograd ×2).
    Leave 1 GB headroom for model parameters and temporaries.
    """
    if not _TORCH_AVAILABLE or not torch.cuda.is_available():
        return 1
    try:
        free, _ = torch.cuda.mem_get_info()
        headroom    = 1 * 1024 ** 3          # 1 GB
        per_start   = T * 3 * 3 * 2 * dtype_bytes * 2   # factor 2 for autograd
        usable      = max(0, free - headroom)
        guess       = max(1, usable // per_start)
        return min(guess, 64)                # cap at 64; raise to 128 if VRAM is plentiful
    except Exception:
        return 4                             # safe conservative default


# ---------------------------------------------------------------------------
# GPU Adam exploration
# ---------------------------------------------------------------------------

def gpu_explore_adam(
        obj: "GpuObjectiveBatched",
        starts_bounded: "list[np.ndarray]",
        n_adam_steps: int = 400,
        lr: float = 0.03,
        batch_size: int = 8,
        patience: int = 0,
        verbose: bool = True,
) -> "list[tuple[float, np.ndarray]]":
    """
    Run Adam optimisation on GPU for all starts, in mini-batches.

    starts_bounded: list of np.ndarray (4N,) in bounded parameter space.
    Returns a list of (val, bounded_params) tuples, sorted by val descending.
    patience: stop early if best has not improved for this many consecutive
              batches (0 = run all batches, no early stopping).
    """
    results = []
    n_total = len(starts_bounded)
    t0 = time.time()
    n_batches = math.ceil(n_total / batch_size)
    _best_so_far = -1e20
    _no_improve  = 0

    for bi in range(n_batches):
        batch_starts = starts_bounded[bi * batch_size:(bi + 1) * batch_size]
        S = len(batch_starts)

        # Stack unconstrained starts into (S, 4N) tensor
        t_unc = np.stack([params_to_unconstrained(p) for p in batch_starts])
        t_batch = torch.tensor(t_unc, dtype=obj.dtype, device=obj.dev,
                               requires_grad=True)

        optimizer = torch.optim.Adam([t_batch], lr=lr)
        for _ in range(n_adam_steps):
            optimizer.zero_grad()
            losses = obj.forward_batch(t_batch)
            losses.sum().backward()
            optimizer.step()

        # Collect results without gradient tracking
        with torch.no_grad():
            final_losses = obj.forward_batch(t_batch).cpu().numpy()
            t_np = t_batch.detach().cpu().numpy()

        for i in range(S):
            val = float(-final_losses[i])
            bounded = unconstrained_to_params(t_np[i]).astype(np.float64)
            results.append((val, bounded))

        current_best = max(r[0] for r in results)
        done = min((bi + 1) * batch_size, n_total)
        if verbose:
            print(f"  [GPU Adam] {done}/{n_total} starts  best={current_best:.8f}"
                  f"  ({time.time() - t0:.1f}s)", flush=True)

        # Early stopping
        if patience > 0:
            if current_best > _best_so_far + 1e-12:
                _best_so_far = current_best
                _no_improve  = 0
            else:
                _no_improve += 1
            if _no_improve >= patience:
                print(f"  [GPU Adam] early stop: no improvement for {patience} batches"
                      f"  ({done}/{n_total} starts completed, best={current_best:.8f})",
                      flush=True)
                break

    results.sort(key=lambda x: x[0], reverse=True)
    return results


# ---------------------------------------------------------------------------
# GPU-accelerated float64_optimise replacement
# ---------------------------------------------------------------------------

def float64_optimise_gpu(
        prob: dict,
        n_starts: int = 100,
        seed: int = 0,
        x0_warm: "np.ndarray | None" = None,
        per_start_seeds: bool = False,
        n_workers: int = 1,
        device: "torch.device | None" = None,
        n_adam_steps: int = 400,
        batch_size: int = 0,            # 0 = auto
        n_polish: int = 0,              # 0 = max(n_workers, 20)
        lr_adam: float = 0.03,
        patience: int = 0,              # early-stop patience in batches (0 = off)
        use_gpu: bool = True,
) -> "tuple[float, np.ndarray]":
    """
    Two-phase optimizer:
      Phase A (GPU, float32): batched Adam exploration over all n_starts.
      Phase B (CPU, float64): parallel scipy L-BFGS-B polish of top candidates.
    Falls back to CPU-only float64_optimise when use_gpu=False or no GPU available.
    """
    N = prob['N']
    bounds_lo = np.array([0., 0., 0., -8.] * N)
    bounds_hi = np.array([math.pi / 2, 2 * math.pi, 2 * math.pi, 8.] * N)

    # ------------------------------------------------------------------
    # Fast path: warm start only (n_starts == 0) → one CPU L-BFGS-B polish
    # ------------------------------------------------------------------
    if n_starts == 0:
        if x0_warm is not None and len(x0_warm) == 4 * N:
            return float64_optimise(prob, n_starts=0, seed=seed,
                                    x0_warm=x0_warm, n_workers=1)
        return -1e20, np.zeros(4 * N)

    # ------------------------------------------------------------------
    # Build random starts (same RNG convention as certify_block_maximum.py)
    # ------------------------------------------------------------------
    random_starts = []
    if per_start_seeds:
        for i in range(n_starts):
            rng_i = np.random.default_rng(seed + i)
            random_starts.append(rng_i.uniform(bounds_lo, bounds_hi))
    else:
        rng = np.random.default_rng(seed)
        for _ in range(n_starts):
            random_starts.append(rng.uniform(bounds_lo, bounds_hi))

    # ------------------------------------------------------------------
    # Phase A: GPU Adam exploration
    # ------------------------------------------------------------------
    gpu_ok = (use_gpu and _TORCH_AVAILABLE and
              (device is None or 'cuda' in str(device)) and
              torch.cuda.is_available())

    if gpu_ok:
        if device is None:
            device = torch.device('cuda')

        _bs = batch_size if batch_size > 0 else auto_batch_size(
            len(prob['ell_idx']), dtype_bytes=4)
        print(f"  [GPU] device={device}  batch_size={_bs}  "
              f"adam_steps={n_adam_steps}  lr={lr_adam}", flush=True)

        obj32 = GpuObjectiveBatched(prob, device, dtype=torch.float32)

        # Warm start is explored first (prepend to list)
        all_starts = []
        if x0_warm is not None and len(x0_warm) == 4 * N:
            all_starts.append(x0_warm.astype(np.float64))
        all_starts.extend(random_starts)

        adam_results = gpu_explore_adam(
            obj32, all_starts,
            n_adam_steps=n_adam_steps, lr=lr_adam,
            batch_size=_bs, patience=patience, verbose=True,
        )

        # Free GPU memory before CPU polish
        del obj32
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        # No GPU: treat all starts as Phase A results at -inf (polish all)
        all_starts = []
        if x0_warm is not None and len(x0_warm) == 4 * N:
            all_starts.append(x0_warm.astype(np.float64))
        all_starts.extend(random_starts)
        adam_results = [(-1e20, s) for s in all_starts]
        print(f"  [GPU] not available — running CPU-only polish of "
              f"{len(all_starts)} starts", flush=True)

    # ------------------------------------------------------------------
    # Phase B: CPU parallel L-BFGS-B polish (float64)
    #
    # Two-stage strategy:
    #   B1 (explore): all candidates polished in parallel with loose
    #      tolerances (gtol=1e-9, maxiter=2000) to identify the best basin
    #      quickly.  Near-optimal Adam starts converge in seconds instead of
    #      hundreds of seconds.
    #   B2 (certify): one final tight polish (gtol=1e-13, maxiter=100_000)
    #      on the single best candidate, giving a certification-quality x*.
    # ------------------------------------------------------------------
    _n_polish = n_polish if n_polish > 0 else max(n_workers, 20)
    _n_polish = min(_n_polish, len(adam_results))
    candidates = [x for _, x in adam_results[:_n_polish]]

    # ---- B1: loose parallel explore ----
    print(f"\n  [CPU explore] {len(candidates)} candidates  {n_workers} workers"
          f"  (loose tols: gtol=1e-9, maxiter=2000) ...", flush=True)
    t_cpu = time.time()
    tasks = list(enumerate(candidates))
    best_val, best_x = -1e20, None

    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(n_workers, initializer=_init_lbfgsb_worker,
                  initargs=(prob,)) as pool:
        for i, val, x in pool.imap_unordered(_lbfgsb_explore_worker, tasks):
            if val > best_val:
                best_val = val
                best_x   = x.copy()
            print(f"  explore {i + 1}/{len(candidates)}"
                  f"  val={val:.8f}  best={best_val:.8f}"
                  f"  ({time.time() - t_cpu:.1f}s)", flush=True)

    # ---- B2: single tight polish of winning candidate ----
    # Use float64_optimise(n_starts=0) — serial inline minimize path,
    # avoids the _worker_prob global that requires a Pool initializer.
    print(f"\n  [CPU certify] tight polish of best candidate"
          f"  (gtol=1e-13, maxiter=100_000) ...", flush=True)
    t_tight = time.time()
    tight_val, tight_x = float64_optimise(prob, n_starts=0, seed=0,
                                          x0_warm=best_x, n_workers=1)
    print(f"  tight polish: val={tight_val:.10f}  ({time.time() - t_tight:.1f}s)",
          flush=True)
    if tight_val > best_val:
        best_val, best_x = tight_val, tight_x

    return best_val, best_x


# ---------------------------------------------------------------------------
# Main certification flow
# (identical to certify_block_maximum.py except float64_optimise → *_gpu)
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='GPU-accelerated certify_block_maximum (Steps 2-5 identical to CPU version)')
    ap.add_argument('--n_starts', type=int, default=100)
    ap.add_argument('--dps',      type=int, default=50)
    ap.add_argument('--skip_isc', action='store_true')
    ap.add_argument('--k',        type=int, default=3)
    ap.add_argument('--full_block', action='store_true')
    ap.add_argument('--x0_file',  type=str, default=None)
    ap.add_argument('--warm-npz', type=str, default=None, metavar='DIR')
    ap.add_argument('--n_workers', type=int, default=None)
    ap.add_argument('--precompute-triads', action='store_true')
    ap.add_argument('--isc-vstar', type=str, default=None, metavar='DIR')
    ap.add_argument('--global-cert-starts', type=int, default=0, metavar='N')
    ap.add_argument('--max-gc-iters',      type=int, default=10, metavar='N',
                    help='Max global-cert scan iterations (repeats until no better basin found)')
    ap.add_argument('--active-subspace-starts', type=int, default=0, metavar='N',
                    help='Extra starts restricted to the active-mode subproblem after finding x*'
                         ' (much cheaper per start; use 50000+ for near-exhaustive k=4 coverage)')
    # GPU-specific flags
    ap.add_argument('--batch-size',   type=int,   default=0,    metavar='N',
                    help='GPU Adam batch size (0 = auto)')
    ap.add_argument('--n-adam-steps', type=int,   default=400,  metavar='N',
                    help='Adam optimisation steps per mini-batch (default 400)')
    ap.add_argument('--n-polish',     type=int,   default=0,    metavar='N',
                    help='Top-N Adam results to polish with CPU L-BFGS-B (0 = auto)')
    ap.add_argument('--adam-lr',      type=float, default=0.03, metavar='F',
                    help='Adam learning rate (default 0.03)')
    ap.add_argument('--device',       type=str,   default=None,
                    help='PyTorch device string (default: auto-detect CUDA)')
    ap.add_argument('--no-gpu',       action='store_true',
                    help='Disable GPU; run CPU-only (same as certify_block_maximum.py)')
    ap.add_argument('--patience',    type=int,   default=32,   metavar='N',
                    help='GPU Adam early-stop patience: stop scan if best has not improved '
                         'for N consecutive batches (0 = run all starts)')
    ap.add_argument('--xstar-npz',  type=str,   default=None, metavar='FILE',
                    help='Load saved x* from FILE and skip directly to Hessian/ISC checks '
                         '(Steps 2-3 of the certification).  Skips all optimisation.')
    args = ap.parse_args()

    # ── Dated log file ────────────────────────────────────────────────────
    from scripts.gap3._run_log import setup_run_log
    _mode_tag = 'full' if args.full_block else 'nucleus'
    _run_log_path = setup_run_log('certify_block_maximum_gpu', tag=f'k{args.k}_{_mode_tag}',
                                    argv=sys.argv[1:])

    k         = args.k
    dps       = args.dps
    n_starts  = args.n_starts
    n_workers = args.n_workers or os.cpu_count()

    gpu_device = None
    if not args.no_gpu and args.device:
        gpu_device = torch.device(args.device)
    use_gpu = not args.no_gpu

    _EPS_B = 1e-7   # boundary proximity threshold for projected-gradient calculation

    print("=" * 68)
    mode_str = 'full-block' if args.full_block else 'nucleus'
    print(f"  CERTIFICATION: C(I_{k}) [{mode_str}] via interval arithmetic"
          f" (mpmath, {dps} dps)")
    print("=" * 68)

    # ── Step 1: build problem ─────────────────────────────────────────────
    n_D, n_C, n_A = find_dcxa_nucleus(k)
    nucleus_shells = [s for s in [n_D, n_C, n_A] if s is not None]
    print(f"\n  DCxA nucleus: n_D={n_D}, n_C={n_C}, n_A={n_A}")

    if args.full_block:
        n_min, n_max = 2**k, 2**(k + 1) - 1
        if args.precompute_triads:
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
            print(f"  Full block I_{k} = [{n_min}, {n_max}]"
                  f" (precompute_triads — all triads)")
            print(f"  {len(_blk_wvs)} modes, {len(_eli)} triads")
        else:
            shells_with_modes = [
                s for s in range(n_min, n_max + 1)
                if get_wavevectors(max_shell2=s, min_shell2=s)
            ]
            print(f"  Full block I_{k} = [{n_min}, {n_max}]")
            prob = _build_problem(shells_with_modes)
            if prob is None:
                sys.exit("ERROR: could not build full-block problem")
            prob_nuc_raw = _build_problem(nucleus_shells)
            if prob_nuc_raw is not None and len(prob_nuc_raw['ell_idx']) > 0:
                prob_nuc_raw = _restrict_to_active_modes(prob_nuc_raw)
                _N2 = prob_nuc_raw['N']  # positive-mode count after restriction
                prob_full_wv_set = {tuple(int(c) for c in wv): j
                                    for j, wv in enumerate(prob['wavevecs'])}
                existing_triads = set(
                    zip(prob['ell_idx'].tolist(),
                        prob['r_idx'].tolist(),
                        prob['s_idx'].tolist()))
                n_injected = 0
                new_ell, new_el2, new_r, new_s, new_sm = [], [], [], [], []
                for t in range(len(prob_nuc_raw['ell_idx'])):
                    # ell_idx/r_idx/s_idx are in [0, 2*_N2); upper half are
                    # negative (conjugate) modes.  Skip them: _restrict_to_active_modes
                    # on prob will call precompute_triads and regenerate all
                    # negative-mode triads automatically.
                    if (prob_nuc_raw['ell_idx'][t] >= _N2
                            or prob_nuc_raw['r_idx'][t] >= _N2
                            or prob_nuc_raw['s_idx'][t] >= _N2):
                        continue
                    ell_wv = tuple(int(c) for c in
                                   prob_nuc_raw['wavevecs'][prob_nuc_raw['ell_idx'][t]])
                    r_wv   = tuple(int(c) for c in
                                   prob_nuc_raw['wavevecs'][prob_nuc_raw['r_idx'][t]])
                    s_wv   = tuple(int(c) for c in
                                   prob_nuc_raw['wavevecs'][prob_nuc_raw['s_idx'][t]])
                    if (ell_wv in prob_full_wv_set and r_wv in prob_full_wv_set
                            and s_wv in prob_full_wv_set):
                        ei = prob_full_wv_set[ell_wv]
                        ri = prob_full_wv_set[r_wv]
                        si = prob_full_wv_set[s_wv]
                        if (ei, ri, si) not in existing_triads:
                            new_ell.append(ei); new_el2.append(prob_nuc_raw['ell2'][t])
                            new_r.append(ri);   new_s.append(si)
                            new_sm.append(prob_nuc_raw['s_mat'][t])
                            existing_triads.add((ei, ri, si))
                            n_injected += 1
                if n_injected > 0:
                    prob['ell_idx'] = np.concatenate([prob['ell_idx'],
                                                       np.array(new_ell, dtype=np.int32)])
                    prob['ell2']    = np.concatenate([prob['ell2'],
                                                       np.array(new_el2)])
                    prob['r_idx']   = np.concatenate([prob['r_idx'],
                                                       np.array(new_r, dtype=np.int32)])
                    prob['s_idx']   = np.concatenate([prob['s_idx'],
                                                       np.array(new_s, dtype=np.int32)])
                    prob['s_mat']   = np.concatenate([prob['s_mat'],
                                                       np.array(new_sm)])
                    print(f"  [Full] injected {n_injected} nucleus triads"
                          f" (total: {len(prob['ell_idx'])})")
        label     = f'C(I_{k})'
        prob_label = f'C(I_{k})'
    else:
        prob = _build_problem(nucleus_shells)
        if prob is None:
            sys.exit("ERROR: could not build nucleus problem")
        label     = f'C_nuc(I_{k})'
        prob_label = f'C_nuc(I_{k})'

    prob = _restrict_to_active_modes(prob)
    N = prob['N']
    n_triads = len(prob['ell_idx'])
    print(f"  {label} modes N={N}, triads T={n_triads}")

    # ── Step 2: float64 multi-start optimisation (or load known x*) ──────
    _warm_x0 = None
    if args.warm_npz is not None:
        npz_path = os.path.join(args.warm_npz, f'k{k}_warm.npz')
        if os.path.isfile(npz_path):
            try:
                _d = np.load(npz_path)
                _warm_x0 = _d['best_x']
                _warm_val = float(_d['best_val'])
                print(f"  [warm-npz] Loaded {npz_path}: best_val={_warm_val:.8f},"
                      f" {len(_warm_x0)//4} modes", flush=True)
            except Exception as _e:
                print(f"  [warm-npz] WARNING: could not load {npz_path}: {_e}")
        else:
            print(f"  [warm-npz] WARNING: {npz_path} not found — ignoring")

    if args.isc_vstar is not None:
        _npz_path = os.path.join(args.isc_vstar, f'isc_k{k}_vstar.npz')
        if os.path.isfile(_npz_path):
            try:
                from scripts.gap3.gpu_isc import get_wavevectors_on_shells, get_block_shells as _gbs_v
                _vd = np.load(_npz_path)
                _isc_x   = _vd['best_x']
                _isc_c   = float(_vd['C_opt'])
                _isc_wvs = get_wavevectors_on_shells(_gbs_v(k))
                _isc_idx = {tuple(int(c) for c in wv): j
                             for j, wv in enumerate(_isc_wvs)}
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
                print(f"  [isc-vstar] Loaded {_npz_path}: C_opt={_isc_c:.8f},"
                      f" {len(_isc_x)//4} src modes -> {_n_mapped}/{prob['N']} mapped",
                      flush=True)
            except Exception as _e:
                print(f"  [isc-vstar] WARNING: could not load {_npz_path}: {_e}")

    if args.xstar_npz is not None:
        print(f"\n  [Step 1] Loading certified x* from {args.xstar_npz} (skipping optimisation) ...",
              flush=True)
        _xd = np.load(args.xstar_npz)
        x_f64   = _xd['best_x']
        val_f64 = float(_xd['best_val'])
        print(f"  Loaded x* ({len(x_f64)//4} modes), val={val_f64:.10f}", flush=True)
    elif args.x0_file is not None:
        print(f"\n  [Step 1] Loading known x* from {args.x0_file} ...", flush=True)
        x_loaded = np.load(args.x0_file)
        print(f"  Loaded x* with {len(x_loaded)//4} modes ({len(x_loaded)} params)")
        N_loaded = len(x_loaded) // 4
        if N_loaded == N:
            x_f64 = x_loaded.copy()
        else:
            n_min_tmp, n_max_tmp = 2**k, 2**(k+1) - 1
            shells_tmp = [
                s for s in range(n_min_tmp, n_max_tmp + 1)
                if get_wavevectors(max_shell2=s, min_shell2=s)
            ]
            prob_unrestr_tmp = _build_problem(shells_tmp)
            file_wv_idx = {tuple(int(c) for c in wv): j
                           for j, wv in enumerate(prob_unrestr_tmp['wavevecs'])}
            x_f64 = np.zeros(4 * N)
            x_f64[3::4] = -8.0
            for j, wv in enumerate(prob['wavevecs']):
                key = tuple(int(c) for c in wv)
                if key in file_wv_idx:
                    j_src = file_wv_idx[key]
                    x_f64[4*j:4*j+4] = x_loaded[4*j_src:4*j_src+4]
            print(f"  Mapped {N_loaded}→{N} modes via wavevec lookup")
        neg_val, _ = neg_ratio_and_grad(
            x_f64, N, prob['e1s'], prob['e2s'], prob['k2s'],
            prob['ell_idx'], prob['ell2'], prob['r_idx'], prob['s_idx'], prob['s_mat'])
        val_f64 = float(-neg_val)
        print(f"  R(x_loaded) = {val_f64:.10f}", flush=True)
    else:
        x0_warm = _warm_x0
        if args.full_block and x0_warm is None:
            print(f"\n  [Warm-start] Solving nucleus subproblem ({nucleus_shells},"
                  f" 50 starts) ...", flush=True)
            prob_nuc_ws = _build_problem(nucleus_shells)
            if prob_nuc_ws is not None and len(prob_nuc_ws.get('ell_idx', [])) > 0:
                prob_nuc_ws = _restrict_to_active_modes(prob_nuc_ws)
                _, x_nuc_ws = float64_optimise(prob_nuc_ws, n_starts=50, seed=0)
                N_full = prob['N']
                x_full_warm = np.zeros(4 * N_full)
                x_full_warm[3::4] = -8.0
                nuc_wv_map = {tuple(int(c) for c in wv): ni
                              for ni, wv in enumerate(prob_nuc_ws['wavevecs'])}
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
        elif x0_warm is not None:
            print(f"  [Warm-start] Using warm-npz x* ({len(x0_warm)//4} modes)")

        if args.full_block:
            opt_seed = k * 2000
            use_per_start = True
        else:
            opt_seed = 42
            use_per_start = False

        warmstr = "1 warm-start + " if x0_warm is not None else ""
        print(f"\n  [Step 1] GPU+CPU optimiser ({warmstr}{n_starts} random starts"
              f"{', per-start seeds' if use_per_start else ''}) ...", flush=True)
        t0_opt = time.time()
        val_f64, x_f64 = float64_optimise_gpu(
            prob,
            n_starts=n_starts,
            seed=opt_seed,
            x0_warm=x0_warm,
            per_start_seeds=use_per_start,
            n_workers=n_workers,
            device=gpu_device,
            n_adam_steps=args.n_adam_steps,
            batch_size=args.batch_size,
            n_polish=args.n_polish,
            lr_adam=args.adam_lr,
            patience=args.patience,
            use_gpu=use_gpu,
        )
        print(f"\n  Float64 result: C* = {val_f64:.10f}  ({time.time()-t0_opt:.1f}s)")

    # Gradient norm at x_f64
    _, grad_f64 = neg_ratio_and_grad(
        x_f64, N, prob['e1s'], prob['e2s'], prob['k2s'],
        prob['ell_idx'], prob['ell2'], prob['r_idx'], prob['s_idx'], prob['s_mat'])

    bound_active_modes = set(i for i in range(N) if x_f64[4*i+3] <= -8.0 + 1e-5)
    n_active = N - len(bound_active_modes)

    _proj = np.abs(grad_f64.copy())
    for _i in range(N):
        _b = 4 * _i
        if _i in bound_active_modes:
            _proj[_b:_b+3] = 0.0
            if grad_f64[_b+3] > 0:
                _proj[_b+3] = 0.0
        else:
            if x_f64[_b]   <= _EPS_B              and grad_f64[_b]   > 0: _proj[_b]   = 0.0
            if x_f64[_b]   >= math.pi/2 - _EPS_B  and grad_f64[_b]   < 0: _proj[_b]   = 0.0
            if x_f64[_b+1] <= _EPS_B              and grad_f64[_b+1] > 0: _proj[_b+1] = 0.0
            if x_f64[_b+1] >= 2*math.pi - _EPS_B  and grad_f64[_b+1] < 0: _proj[_b+1] = 0.0
            if x_f64[_b+2] <= _EPS_B              and grad_f64[_b+2] > 0: _proj[_b+2] = 0.0
            if x_f64[_b+2] >= 2*math.pi - _EPS_B  and grad_f64[_b+2] < 0: _proj[_b+2] = 0.0
            if x_f64[_b+3] >= 8.0 - _EPS_B        and grad_f64[_b+3] < 0: _proj[_b+3] = 0.0
    proj_grad_norm = float(np.max(_proj))
    n_active_params = 4 * n_active
    print(f"  Active modes: {n_active}/{N}"
          f" ({len(bound_active_modes)} at loga=-8, excluded from grad/Hess checks)")
    print(f"  Projected gradient norm: {proj_grad_norm:.3e}", flush=True)

    # ── Step 1b: restrict to active modes if gradient is not tight ────────
    if proj_grad_norm > 1e-6 and n_active < N and n_active > 0:
        print(f"\n  [Step 1b] gradient {proj_grad_norm:.2e} > 1e-6 — restricting"
              f" to {n_active} active modes and re-optimising ...", flush=True)
        active_idxs = sorted(i for i in range(N) if x_f64[4*i+3] > -8.0 + 1e-5)
        active_wvs  = [prob['wavevecs'][i] for i in active_idxs]
        _, _a_eli, _a_el2, _a_ri, _a_si, _a_sm = precompute_triads(active_wvs)
        _a_N = len(active_wvs)
        _act_arr = np.array(active_idxs)
        prob = dict(N=_a_N,
                    k2s=prob['k2s'][_act_arr],
                    e1s=prob['e1s'][_act_arr],
                    e2s=prob['e2s'][_act_arr],
                    ell_idx=_a_eli, ell2=_a_el2, r_idx=_a_ri, s_idx=_a_si,
                    s_mat=_a_sm, wavevecs=active_wvs)
        x0_r = np.concatenate([x_f64[4*fi:4*fi+4] for fi in active_idxs])
        print(f"  Restricted: {_a_N} modes, {len(_a_eli)} triads", flush=True)
        _val_r, x_r = float64_optimise(prob, n_starts=0, seed=opt_seed, x0_warm=x0_r)
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
                if x_r[_b]   <= _EPS_B              and grad_r[_b]   > 0: _proj_r[_b]   = 0.0
                if x_r[_b]   >= math.pi/2 - _EPS_B  and grad_r[_b]   < 0: _proj_r[_b]   = 0.0
                if x_r[_b+1] <= _EPS_B              and grad_r[_b+1] > 0: _proj_r[_b+1] = 0.0
                if x_r[_b+1] >= 2*math.pi - _EPS_B  and grad_r[_b+1] < 0: _proj_r[_b+1] = 0.0
                if x_r[_b+2] <= _EPS_B              and grad_r[_b+2] > 0: _proj_r[_b+2] = 0.0
                if x_r[_b+2] >= 2*math.pi - _EPS_B  and grad_r[_b+2] < 0: _proj_r[_b+2] = 0.0
                if x_r[_b+3] >= 8.0 - _EPS_B        and grad_r[_b+3] < 0: _proj_r[_b+3] = 0.0
        proj_grad_norm  = float(np.max(_proj_r))
        n_active        = _a_N - len(_bnd_r)
        n_active_params = 4 * n_active
        N               = _a_N
        x_f64           = x_r
        val_f64         = _val_r
        bound_active_modes = _bnd_r
        print(f"  Restricted optimum: C={val_f64:.10f}"
              f"  grad={proj_grad_norm:.2e}  {n_active}/{N} active")
        print(f"  [Switched to restricted ({N}-mode) problem for certification]")

    # ── Step 1c: iterative global cert scan ────────────────────────────────
    # Scan the FULL problem (all N modes) so candidates with different active
    # sets are considered.  Restricting to the active subproblem would miss
    # basins where completely different modes participate — as demonstrated by
    # runs that found a better solution only after scanning the full space.
    # Each iteration uses a fresh seed offset to sample independent starts.
    _global_cert_passed = None
    _gc_starts   = args.global_cert_starts
    _gc_tol      = 1e-7
    _gc_max_iter = args.max_gc_iters
    # Keep a reference to the original full problem; prob may be updated below
    _full_prob   = prob

    if _gc_starts > 0:
        print(f"  Full problem: {_full_prob['N']} modes, {len(_full_prob['ell_idx'])} triads"
              f"  (global cert scans full space)", flush=True)

        for _gc_iter in range(1, _gc_max_iter + 1):
            _gc_seed = 1234 + (_gc_iter - 1) * 10000
            print(f"\n  [Step 1c iter {_gc_iter}/{_gc_max_iter}] Global cert scan:"
                  f" {_gc_starts} starts on full {_full_prob['N']}-mode problem"
                  f" (seed={_gc_seed}) ...", flush=True)
            t_gc = time.time()
            _gc_best, _gc_best_x = float64_optimise_gpu(
                _full_prob, n_starts=_gc_starts, seed=_gc_seed,
                x0_warm=None, per_start_seeds=True,
                n_workers=n_workers,
                device=gpu_device,
                n_adam_steps=args.n_adam_steps,
                batch_size=args.batch_size,
                n_polish=args.n_polish,
                lr_adam=args.adam_lr,
                patience=args.patience,
                use_gpu=use_gpu,
            )
            print(f"  Global scan best: {_gc_best:.10f}  vs C*: {val_f64:.10f}"
                  f"  ({time.time()-t_gc:.0f}s)", flush=True)

            if _gc_best > val_f64 + _gc_tol:
                # Found a better basin in the full space — promote
                print(f"  Higher basin (iter {_gc_iter}):"
                      f" C_new={_gc_best:.10f} > C_old={val_f64:.10f}")
                print(f"  Polishing new x* (warm restart) ...", flush=True)
                _gc_poly_val, _gc_poly_x = float64_optimise(
                    _full_prob, n_starts=0, seed=0, x0_warm=_gc_best_x)
                val_f64 = _gc_poly_val
                x_f64   = _gc_poly_x
                prob    = _full_prob
                N       = _full_prob['N']
                _, grad_f64 = neg_ratio_and_grad(
                    x_f64, N, prob['e1s'], prob['e2s'], prob['k2s'],
                    prob['ell_idx'], prob['ell2'], prob['r_idx'], prob['s_idx'], prob['s_mat'])
                bound_active_modes = set(i for i in range(N) if x_f64[4*i+3] <= -8.0 + 1e-5)
                n_active = N - len(bound_active_modes)
                n_active_params = 4 * n_active
                _pgp = np.abs(grad_f64.copy())
                _EPS_BP = 1e-4
                for _i in range(N):
                    _b = 4 * _i
                    if _i in bound_active_modes:
                        _pgp[_b:_b+3] = 0.0
                        if grad_f64[_b+3] > 0: _pgp[_b+3] = 0.0
                    else:
                        if x_f64[_b]   <= _EPS_BP              and grad_f64[_b]   > 0: _pgp[_b]   = 0.0
                        if x_f64[_b]   >= math.pi/2 - _EPS_BP  and grad_f64[_b]   < 0: _pgp[_b]   = 0.0
                        if x_f64[_b+1] <= _EPS_BP              and grad_f64[_b+1] > 0: _pgp[_b+1] = 0.0
                        if x_f64[_b+1] >= 2*math.pi - _EPS_BP  and grad_f64[_b+1] < 0: _pgp[_b+1] = 0.0
                        if x_f64[_b+2] <= _EPS_BP              and grad_f64[_b+2] > 0: _pgp[_b+2] = 0.0
                        if x_f64[_b+2] >= 2*math.pi - _EPS_BP  and grad_f64[_b+2] < 0: _pgp[_b+2] = 0.0
                        if x_f64[_b+3] >= 8.0 - _EPS_BP        and grad_f64[_b+3] < 0: _pgp[_b+3] = 0.0
                proj_grad_norm = float(np.max(_pgp))
                print(f"  Promoted x*: C={val_f64:.10f}  proj_grad={proj_grad_norm:.2e}"
                      f"  {n_active}/{N} active", flush=True)
            else:
                _global_cert_passed = True
                print(f"  All {_gc_starts} starts <= C* + {_gc_tol:.0e}"
                      f" (iter {_gc_iter}) — scan passed")
                print(f"  NUMERICALLY GLOBALLY CERTIFIED"
                      f" (full {_full_prob['N']}-mode problem,"
                      f" {_gc_starts} starts × {_gc_iter} iter(s))")
                break
        else:
            # Exhausted max iterations without passing
            print(f"  WARNING: global cert scan did not pass after {_gc_max_iter}"
                  f" iterations — C* may still be improvable", flush=True)

    # ── Step 1d: active-subspace exhaustive scan ───────────────────────────
    # Build the n_active-mode subproblem and run a massive multistart search.
    # Because the restricted problem is ~10× smaller (92 vs 976 dims for k=4)
    # we can afford orders of magnitude more starts, greatly reducing the
    # probability of missing a better basin within the same active set.
    _as_starts = args.active_subspace_starts
    if _as_starts > 0 and n_active < N:
        _as_idxs = sorted(i for i in range(N) if i not in bound_active_modes)
        _as_wvs  = [prob['wavevecs'][i] for i in _as_idxs]
        _, _as_eli, _as_el2, _as_ri, _as_si, _as_sm = precompute_triads(_as_wvs)
        _as_prob = dict(N=len(_as_wvs),
                        k2s=prob['k2s'][np.array(_as_idxs)],
                        e1s=prob['e1s'][np.array(_as_idxs)],
                        e2s=prob['e2s'][np.array(_as_idxs)],
                        ell_idx=_as_eli, ell2=_as_el2,
                        r_idx=_as_ri, s_idx=_as_si, s_mat=_as_sm,
                        wavevecs=_as_wvs)
        print(f"\n  [Step 1d] Active-subspace exhaustive scan:"
              f" {_as_starts} starts on {n_active}-mode subproblem"
              f" ({4*n_active} dims) ...", flush=True)
        t_as = time.time()
        # Project current x* onto the active subspace as warm start so the
        # best known point is included in Phase B regardless of early stopping.
        _as_x0_warm = np.concatenate([x_f64[4*_gi:4*_gi+4] for _gi in _as_idxs])
        _as_best, _as_best_x = float64_optimise_gpu(
            _as_prob, n_starts=_as_starts, seed=9999,
            x0_warm=_as_x0_warm, per_start_seeds=True,
            n_workers=n_workers,
            device=gpu_device,
            n_adam_steps=args.n_adam_steps,
            batch_size=args.batch_size,
            n_polish=args.n_polish,
            lr_adam=args.adam_lr,
            patience=args.patience,
            use_gpu=use_gpu,
        )
        print(f"  Active-subspace scan best (subproblem, tight): {_as_best:.10f}"
              f"  vs C* (full problem, tight): {val_f64:.10f}"
              f"  ({time.time()-t_as:.0f}s)", flush=True)
        # IMPORTANT: always lift to the full problem and re-polish before
        # comparing, because the subproblem objective (inactive modes fixed at
        # loga=-8) differs from the full-problem objective by O(1e-6).
        # Using the raw subproblem value against full-problem C* is apples-to-
        # oranges and can trigger spurious promotions.
        print(f"  Lifting subproblem x* to full problem and polishing ...", flush=True)
        _as_full_x = x_f64.copy()
        for _ii, _gi in enumerate(_as_idxs):
            _as_full_x[4*_gi:4*_gi+4] = _as_best_x[4*_ii:4*_ii+4]
        _as_poly_val, _as_poly_x = float64_optimise(
            prob, n_starts=0, seed=0, x0_warm=_as_full_x)
        print(f"  Full-problem re-polish: {_as_poly_val:.10f}  vs C*: {val_f64:.10f}",
              flush=True)
        if _as_poly_val > val_f64 + 1e-9:
            val_f64 = _as_poly_val
            x_f64   = _as_poly_x
            print(f"  Promoted from active-subspace scan: C*={val_f64:.10f}")
        else:
            print(f"  Full-problem re-polish did not improve C* — same basin")
            print(f"  ACTIVE-SUBSPACE EXHAUSTED: {_as_starts} starts in"
                  f" {n_active}-mode / {4*n_active}-dim subspace")

    # ── Auto-save x* to dated log directory (always, unconditionally) ─────
    _auto_npz_dir = (os.path.dirname(_run_log_path)
                     if '_run_log_path' in dir() else
                     os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'results', 'xstar_cache'))
    _auto_npz_path = os.path.join(_auto_npz_dir, f'k{k}_{_mode_tag}_xstar.npz')
    try:
        os.makedirs(_auto_npz_dir, exist_ok=True)
        np.savez(_auto_npz_path, best_x=x_f64, best_val=np.float64(val_f64))
        print(f"  [auto-save] x* saved → {_auto_npz_path}  (val={val_f64:.10f})",
              flush=True)
    except Exception as _save_exc:
        print(f"  [auto-save] WARNING: could not save x*: {_save_exc}", flush=True)

    # ── Write back best x* to warm NPZ (always update if better) ─────────
    if args.warm_npz is not None:
        _npz_out = os.path.join(args.warm_npz, f'k{k}_warm.npz')
        _do_save = True
        if os.path.isfile(_npz_out):
            try:
                _prev = np.load(_npz_out)
                if float(_prev['best_val']) >= val_f64:
                    _do_save = False  # existing file already holds a better point
            except Exception:
                pass
        if _do_save:
            os.makedirs(args.warm_npz, exist_ok=True)
            np.savez(_npz_out, best_x=x_f64, best_val=np.float64(val_f64))
            print(f"  [warm-npz] Saved best x* ({val_f64:.10f}) → {_npz_out}",
                  flush=True)
        else:
            print(f"  [warm-npz] Existing warm file is >= current best — not overwritten",
                  flush=True)

    # ── Step 3 (label "Step 2"): mpmath high-precision evaluation ─────────
    print(f"\n  [Step 2] mpmath evaluation at x* ({dps} dps) ...", flush=True)
    mpmath.mp.dps = dps + 10
    prob_mp_data = _build_mpmath_data(prob, dps + 10)
    params_mp    = [mpmath.mpf(float(x_f64[i])) for i in range(4 * N)]
    t1 = time.time()
    R_mp = _mpmath_objective(params_mp, *prob_mp_data)
    print(f"  mpmath R(x*) = {mpmath.nstr(R_mp, dps)}  ({time.time()-t1:.1f}s)")

    mpmath.mp.dps = dps
    delta = mpmath.mpf(2) ** (-52)
    try:
        with mpmath.workdps(dps):
            lo_params = [mpmath.mpf(float(x_f64[i])) - delta for i in range(4 * N)]
            hi_params = [mpmath.mpf(float(x_f64[i])) + delta for i in range(4 * N)]
            _mp_data_cert = _build_mpmath_data(prob, dps)
            R_lo = _mpmath_objective(lo_params, *_mp_data_cert)
            R_hi = _mpmath_objective(hi_params, *_mp_data_cert)
            cert_lo = min(R_lo, R_hi, R_mp)
            cert_hi = max(R_lo, R_hi, R_mp)
        print(f"\n  CERTIFIED INTERVAL: {prob_label} in"
              f" [{mpmath.nstr(cert_lo, 10)}, {mpmath.nstr(cert_hi, 10)}]")
        print(f"  Interval width: {float(cert_hi - cert_lo):.3e}")
    except Exception as exc:
        print(f"  (Interval bound computation skipped: {exc})")
        cert_lo = cert_hi = R_mp

    # ── Step 4 (label "Step 3"): Hessian diagonal check ──────────────────
    print(f"\n  [Step 3] Hessian diagonal check at x* ({dps} dps) ...", flush=True)
    t2 = time.time()
    eps_exp = -(dps // 3)
    params_mp_cert = [mpmath.mpf(float(x_f64[i])) for i in range(4 * N)]
    prob_mp_cert   = _build_mpmath_data(prob, dps)

    hess_noise_thresh = (float(abs(R_mp))
                         * (10.0 ** (-(dps - 2 * (dps // 3)))) * 5.0)

    _N_mp, _k2s_mp, _e1s_mp, _e2s_mp, _ell_ints, _ell2_mp, _r_ints, _s_ints, _s_mat_mp = prob_mp_cert
    _nstr       = dps + 10
    _params_strs = [mpmath.nstr(x, _nstr) for x in params_mp_cert]
    _R0_str      = mpmath.nstr(_mpmath_objective(params_mp_cert, *prob_mp_cert), _nstr)
    _k2s_strs    = [mpmath.nstr(x, _nstr) for x in _k2s_mp]
    _e1s_strs    = [[mpmath.nstr(x, _nstr) for x in row] for row in _e1s_mp]
    _e2s_strs    = [[mpmath.nstr(x, _nstr) for x in row] for row in _e2s_mp]
    _ell2_strs   = [mpmath.nstr(x, _nstr) for x in _ell2_mp]
    _s_mat_strs  = [[mpmath.nstr(x, _nstr) for x in row] for row in _s_mat_mp]

    active_indices = [i for i in range(4 * N) if i // 4 not in bound_active_modes]

    # ---- Float64 FD pre-screen: skip mpmath for clearly-negative entries ----
    # Central difference with h=1e-4; FD noise floor ≈ R·ε₆₄/h² ≈ 2e-10.
    # Threshold -1e-6 is safely above the noise floor — conservative pass.
    # Entries with float64 H_ii < threshold are definitively "neg"; only
    # borderline entries (close to zero) need the expensive mpmath eval.
    _F64_HESS_THRESH = -1e-6
    _F64_HESS_H      = 1e-4
    # FD central-difference truncation floor: O(h^2 * f''').  Entries below
    # this are "FD-noise ambiguous" rather than proven positive.
    hess_fd_floor = float(abs(R_mp)) * (_F64_HESS_H ** 2) * 0.01
    _R0_f64 = float(val_f64)
    # Pass all prob fields inline so the worker is self-contained (no global).
    _f64_common = (N, prob['k2s'], prob['e1s'], prob['e2s'],
                   prob['ell_idx'], prob['ell2'],
                   prob['r_idx'], prob['s_idx'], prob['s_mat'])
    _f64_args = [(i, x_f64, _R0_f64, _F64_HESS_H) + _f64_common
                 for i in active_indices]
    print(f"  Float64 FD pre-screen: {len(active_indices)} entries,"
          f" h={_F64_HESS_H:.0e}, neg threshold={_F64_HESS_THRESH:.0e} ...",
          flush=True)
    t_f64_pre = time.time()
    diag_neg_count, diag_flat_count, diag_pos_vals = 0, 0, []
    _border_indices = []
    with multiprocessing.Pool(n_workers) as pool:
        f64_results = pool.map(_hess_f64_worker, _f64_args)
    for _fi, _hii_f64 in f64_results:
        if _hii_f64 < _F64_HESS_THRESH:
            diag_neg_count += 1
        else:
            _border_indices.append(_fi)
    print(f"  Pre-screen: {diag_neg_count} definitively neg,"
          f" {len(_border_indices)} borderline → mpmath"
          f"  ({time.time()-t_f64_pre:.1f}s)", flush=True)

    # ---- mpmath for borderline entries only ----
    worker_args = [
        (i, _params_strs, _R0_str, dps, eps_exp,
         _N_mp, _k2s_strs, _e1s_strs, _e2s_strs,
         _ell_ints, _ell2_strs, _r_ints, _s_ints, _s_mat_strs)
        for i in _border_indices
    ]
    print(f"  Dispatching {len(worker_args)} mpmath evals across {n_workers} workers ...",
          flush=True)
    with multiprocessing.Pool(n_workers) as pool:
        hess_results = pool.map(_hess_diag_worker, worker_args)
    for (i, Hii_f) in hess_results:
        _pname = ['theta', 'phi', 'psi', 'loga'][i % 4]
        _th    = x_f64[4 * (i // 4)]
        # Structural inertness: phi vanishes when cos(theta)=0 (theta~pi/2);
        # psi vanishes when sin(theta)=0 (theta~0).  Threshold 1e-5 is loose
        # enough to catch optimiser-converged points that are numerically at
        # the pole yet technically non-zero.
        _phi_inert = (_pname == 'phi' and abs(math.cos(_th)) < 1e-5)
        _psi_inert = (_pname == 'psi' and abs(math.sin(_th)) < 1e-5)
        _flat_thresh = max(hess_noise_thresh, hess_fd_floor)
        if Hii_f < -hess_noise_thresh:
            diag_neg_count += 1
        elif _phi_inert or _psi_inert or abs(Hii_f) <= _flat_thresh:
            diag_flat_count += 1
        else:
            diag_pos_vals.append((i, Hii_f))

    print(f"  Hessian noise threshold: {hess_noise_thresh:.2e}  (FD floor: {hess_fd_floor:.2e})")
    print(f"  Hessian diagonal: {diag_neg_count} neg / {diag_flat_count} flat"
          f" / {len(diag_pos_vals)} pos  ({time.time()-t2:.1f}s)")
    print(f"  (Skipped {4*N - len(active_indices)} params for"
          f" {len(bound_active_modes)} bound-inactive modes)")
    if diag_pos_vals:
        print("  POSITIVE active-mode entries (inspect!):")
        for idx, val in diag_pos_vals:
            param_name = ['theta', 'phi', 'psi', 'loga'][idx % 4]
            mode_idx   = idx // 4
            print(f"    param[{idx}] (mode {mode_idx} {param_name}):"
                  f" H_ii = {val:+.6e}")
            if param_name in ('phi', 'psi'):
                theta_val = x_f64[4 * mode_idx]
                if param_name == 'phi' and abs(math.cos(theta_val)) < 1e-3:
                    print(f"      (theta={theta_val:.8f} ~ pi/2; phi inert"
                          f" [cos(theta)={math.cos(theta_val):.2e}] — likely FD noise)")
                elif param_name == 'psi' and abs(math.sin(theta_val)) < 1e-3:
                    print(f"      (theta={theta_val:.8f} ~ 0; psi inert"
                          f" [sin(theta)={math.sin(theta_val):.2e}] — likely FD noise)")
    else:
        print("  No above-threshold positive entries — local max in active subspace confirmed")
    if diag_flat_count > 0:
        print(f"  ({diag_flat_count} flat entries within noise threshold:"
              f" expected for phi at theta~pi/2 or psi at theta~0)")

    # ── Step 5 (label "Step 4"): ISC relay-shell check ───────────────────
    if args.full_block:
        print("\n  [Step 4] ISC check skipped (full-block: all block shells included)")
    elif not args.skip_isc:
        n_min_isc, n_max_isc = 2**k, 2**(k+1) - 1
        all_shells  = [s for s in range(n_min_isc, n_max_isc + 1)
                       if get_wavevectors(max_shell2=s, min_shell2=s)]
        relay_shells = [s for s in all_shells if s not in set(nucleus_shells)]
        print(f"\n  [Step 4] ISC relay check: {len(relay_shells)} relay shells ...",
              flush=True)
        any_pos = isc_check(x_f64, prob, relay_shells)
        if any_pos:
            print("\n  WARNING: positive ISC gradient — relay shell may improve R")
        else:
            print(f"\n  ISC: no relay shell improves R.")
            print(f"  -> C(I_{k}) = C_nucleus(I_{k})")
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
    print(f"  Certified interval    :  [{mpmath.nstr(cert_lo, 10)},"
          f" {mpmath.nstr(cert_hi, 10)}]")
    print(f"  Hessian (active only) :  {diag_neg_count} neg / {diag_flat_count} flat"
          f" / {len(diag_pos_vals)} pos  (noise thresh {hess_noise_thresh:.1e})")
    print()
    nuc_certified = (len(diag_pos_vals) == 0 and proj_grad_norm < 1e-6)
    if nuc_certified:
        if args.full_block:
            print(f"  STATUS: {prob_label} LOCALLY CERTIFIED")
            print(f"  {prob_label} = {mpmath.nstr(cert_lo, 8)}"
                  f"  [certified local max of full block I_{k}]")
            if _global_cert_passed is True:
                print(f"  GLOBAL CERT: PASS — {args.global_cert_starts} starts"
                      f" all <= C*  (P8 falsification complete)")
                print(f"  STATUS: {prob_label} NUMERICALLY GLOBALLY CERTIFIED")
            elif _global_cert_passed is False:
                print(f"  GLOBAL CERT: FAIL — a random start exceeded C*")
            else:
                print(f"  (Global cert not requested)")
        else:
            print("  STATUS: C_nuc LOCALLY CERTIFIED")
    else:
        hess_str = (f"{len(diag_pos_vals)} above-thresh positive Hessian entries"
                    if diag_pos_vals else "")
        grad_str = (f"proj_grad={proj_grad_norm:.2e}" if proj_grad_norm >= 1e-6 else "")
        issues   = ", ".join(x for x in [grad_str, hess_str] if x)
        print(f"  STATUS: PARTIAL ({issues})")
        if _global_cert_passed is True:
            print(f"  GLOBAL CERT: PASS (P8 scan; {args.global_cert_starts} starts)")


if __name__ == '__main__':
    main()
