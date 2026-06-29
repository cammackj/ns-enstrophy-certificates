# K3 Closed-Form Work

Status: research note for Paper 2 integration.  The old scan value is stale and
should no longer be asserted as the exact value of `C(I_3)`.

Date: 2026-05-31

Update: the reduced active theorem and the current full-block bridge audit are
now consolidated in `references/K3_CERTIFICATE_PACKET.md`.  That packet
supersedes the older interval-branching status notes below.

## Executive Summary

The old paper value for the k=3 block was the nucleus-grid scan/certification
value

```text
C(I_3) scan = 0.021936432440143745400090178422268102957066551890974
```

There is also a later April 23 full-block GPU certificate

```text
scripts/gap3/results/2026-04-23/certify_block_maximum_gpu_k3_full_221803.txt
```

which reports

```text
C(I_3) = 0.02193639436737329784036868032691608946216906599417
Hessian (active only): 129 neg / 3 flat / 0 pos
GLOBAL CERT: PASS -- 2000 starts all <= C*
STATUS: C(I_3) NUMERICALLY GLOBALLY CERTIFIED
```

That April 23 run was not partial; it was a valid numerical certificate for the
finite-floor full-block optimization protocol it executed.  It is nevertheless
superseded as the displayed k=3 comparison value because the refined zero-floor
active-set computation below finds a slightly larger algebraic candidate.  This
does not contradict the April 23 global-scan pass: that pass used the criterion
`all starts <= C* + 1e-7`, while the refined algebraic candidate is only about
`7.51e-8` above the April 23 value.

The closed-form reduction work found a slightly larger and more structured algebraic candidate value

```text
C_3 closed-form candidate = 0.021936469459403747249299192478957700397867315103825...
```

The difference is about `3.70e-8`.  The old value should be treated as a
floor/basin miss, not as the exact block value.  The refined algebraic candidate
is still comfortably below the k=2 value

```text
C(I_2) = 0.022741865409341...
```

so it strengthens the k=3 story without threatening the finite-block comparison
with k=2.  In the paper, do not state the stale scan value as an equality.
Instead, use k=3 only for the strict comparison `C(I_3)<C(I_2)`, and record the
degree-40 algebraic candidate as the refined active-set/nucleus certificate.

## Support

The closed-form candidate lives on the 9 positive modes

```text
(2, 2, 0)
(1, 0, -3)
(0, 1, -3)
(1, 2, 3)
(2, 1, 3)
(3, 1, 0)
(1, 3, 0)
(2, -1, -3)
(1, -2, 3)
```

The observed optimizer has coordinate-swap equivariance. In the 8-variable equivariant coordinates, the numerical optimizer is

```text
x1 =  0.3755203516598824
y1 =  0.5413115134892512
x3 =  0.3928935550181440
y3 =  0.3488764904801380
x5 = -0.2213494977484710
y5 = -0.1130251988774326
x7 =  0.03541513479120969
y7 =  0.007650223088043244
```

The tiny A-shell pair is small but not zero; suppressing it gives a nearby 6-variable system, but the released gradient in the suppressed directions is nonzero, so the 8-variable system is the correct stationary candidate.

## Active-Set Verification Update (May 30, 2026)

A follow-up verifier now checks the 9-mode candidate against the full k=3
DCxA nucleus objective while varying the artificial inactive-mode floor.  The
script is

```text
scripts/gap3/k3_active_set_verify.py
```

and the saved run with inactive-release gradient summaries is

```text
results/k3_active_set_verify_20260530_170941.json
```

The main finding is that the old scan-backed value is depressed by the
`loga=-8` inactive-mode floor used by the historical certifier.  Embedding the
9-mode candidate into the full 42-mode nucleus and polishing gives:

| inactive floor | polished value | gap below 9-mode value |
|---:|---:|---:|
| `-8`  | `0.021936426794806556` | `4.266e-8` |
| `-10` | `0.021936459498277522` | `9.961e-9` |
| `-12` | `0.021936468475649906` | `9.834e-10` |
| `-16` | `0.021936469326079608` | `1.330e-10` |
| `-24` | `0.021936469438815144` | `2.025e-11` |
| `-30` | `0.021936469459008651` | `5.156e-14` |

The same floor sweep records the scaled inactive-mode release derivative
`d(-R)/d(exp(loga))`.  By floor `-12` all inactive release coefficients are
positive at the reported tolerance; for floors `-20`, `-24`, and `-30` the
minimum scaled coefficient is approximately `+5.671e-3`.  Thus, in the low-floor
limit, releasing an inactive amplitude decreases `R` to first order.

A second diagnostic minimizes the inactive release coefficient over each
inactive mode's angular/polarization variables:

```text
scripts/gap3/k3_inactive_release_scan.py
results/k3_inactive_release_scan_20260530_171502.json
```

This scan used 128 random angular samples per inactive mode, polished the four
lowest samples, and tested all 33 inactive modes in the full nucleus.  No mode
had a negative one-sided release coefficient.  The worst case was

```text
mode index 2, shell 8, wavevector (2,-2,0)
minimum d(-R)/d(exp(loga)) at release floor -20: +8.914029480754e-3
same minimizing angles checked at floor -24: +5.705481815507e-3
negative coefficients below -1e-10: 0
```

This is still numerical, but it is a stronger KKT-facing check than evaluating
one arbitrary inactive polarization: it searches for the most dangerous
first-order release direction in each inactive mode.

At floor `-30`, a 58-start perturbation scan gave:

```text
target support value: 0.021936469459060207
basin starts: 58
active-support perturbations: 40/40 returned to the candidate within 1e-10
all starts within 1e-8: 40/58
best value: 0.021936469459354489
```

The remaining 18 starts deliberately released many inactive modes with random
tiny amplitudes; they fell into lower basins.  This supports the interpretation
that the higher value is a stable sparse active-set basin, while blind
full-space starts can miss it.  The next certificate should therefore use an
exact-zero active-set/KKT formulation or a low-floor limit, not the old
`loga=-8` value as final evidence.

## Refined Targeted Certificate Update (May 30/31, 2026)

The active seed must be refined from the historical warm state before running
the full-nucleus diagnostics.  Without this refinement the support value is
short by about `3.4e-13`, which is enough to pollute extremely low-floor release
tests.

The refined active-support Hessian check is saved in

```text
results/k3_active_hessian_check_20260530_refined.json
```

and gives

```text
refined active value = 0.021936469459403255
mpmath value         = 0.0219364694594032577498201373909698859566031752181598175047860125864578757816033422186266182
active gradient max  = 7.104e-10
```

For the Hessian of the negative objective `-R` on the 36-parameter active
support, finite-difference steps `1e-4`, `3e-5`, and `1e-5` all give

```text
30 positive / 6 flat / 0 negative.
```

The refined full-nucleus embedding run

```text
results/k3_active_set_verify_20260530_refined.json
```

uses the full 42-mode DCxA nucleus with 1080 triads.  Its floor sweep reaches
the refined active value at low floor:

| inactive floor | polished gap to refined active value | inactive scaled minimum |
|---:|---:|---:|
| `-12` | `+8.852e-10` | `+7.186e-07` |
| `-20` | `+7.846e-12` | `+2.058e-05` |
| `-30` | `+5.106e-14` | `+5.766e-03` |
| `-40` | `-6.939e-18` | `+5.766e-03` |

The seeded basin scan at floor `-40` used 128 starts and had 80 hits within
`1e-10` of the refined target; the best basin value was
`0.021936469459403356`, within `1.1e-16` of the refined target in double
precision.

The refined inactive-release scan

```text
results/k3_inactive_release_scan_20260530_refined.json
```

searched all 33 inactive full-nucleus modes using 2048 angular samples and 12
polish starts per inactive mode.  No negative release coefficient was found.
The worst mode is `(2,-2,0)`, with coefficient

```text
min d(-R)/d(exp(loga)) at floor -24: +9.429049075928e-3.
```

The high-precision mpmath objective-difference check

```text
results/k3_inactive_release_mpmath_check_20260530_refined_floor30.json
```

confirms the same worst release direction remains positive through floor `-30`:

```text
floor -20: +0.0095355299639388727824281761949719979131466162753327
floor -24: +0.0092830507245898466269150925299819871480612447463874
floor -28: +0.0074174674091778287648558295373840929260620991583651
floor -30: +0.0037101368700497626101189990611527095484124921201821
```

The algebraic polynomial for `W=C_3^2` is saved in

```text
results/k3_w_primitive_polynomial.txt
```

It is primitive, irreducible over `ZZ`, degree 40, height 1331 bits, matches
133/133 CRT primes and 3/3 held-out primes, and has exactly one root in the
recorded decimal isolating interval for the displayed `W`.

## Kernel Formula

After imposing the observed phase/sign pattern and coordinate equivariance, the k=3 value reduces to a five-variable kernel

```text
K_3(p,q,r,h,theta)
  = sqrt(70) q [ p (sqrt(P_+) + sqrt(P_-)) + sqrt(2) h r sqrt(Q) ]
    / [ 280 A sqrt(G) ]

A = 2 + 5 p^2 + 7 q^2 + 5 r^2 + 7 h^2
G = 8 + 25 p^2 + 49 q^2 + 25 r^2 + 49 h^2

P_sigma = 1259 - 108 sigma sqrt(35) + 637 r^2
          + 2r [ (162 sigma sqrt(7) - 299 sqrt(5)) cos(theta)
                 - (180 sqrt(2) + 39 sigma sqrt(70)) sin(theta) ]

Q = 575 + 325 cos(2 theta) - 150 sqrt(10) sin(2 theta)
```

The variables `p`, `q`, and `h` can be eliminated analytically, leaving a three-variable maximization in `(t,r,theta)` where

```text
t = sqrt(q^2 + h^2) / p.
```

The script `scripts/gap3/k3_three_variable_reduction.py` implements this reduced objective and recovers the numerical value above.

## Closed Algebraic Root Form

The reduced closed form can be written as a finite algebraic system for

```text
(t, r, c, s, a, b, e, d, ell, w),   w = C_3^2,
```

where `c=cos(theta)`, `s=sin(theta)`, and

```text
Sigma = a+b
alpha = 2 + 5 r^2
beta  = 8 + 25 r^2
A_t   = 5 + 7 t^2
G_t   = 25 + 49 t^2

a^2 = P_+(r,c,s)
b^2 = P_-(r,c,s)
e^2 = Q(c,s)
d^2 = Sigma^2 + 16 r^2 e^2 t^2
ell^2 alpha G_t = alpha G_t + 8 A_t beta.
```

The value equation is

```text
4480 w (d+Sigma) alpha A_t G_t (3+ell)^3
  = t^2 (d+3 Sigma)^3 (1+ell),
```

and the three interior stationarity equations are

```text
d/d(log t) log F = 0,
d/d(log r) log F = 0,
d/d theta  log F = 0,

F = t^2 (d+3 Sigma)^3 (1+ell)
    / [ (d+Sigma) alpha A_t G_t (3+ell)^3 ].
```

After clearing denominators, these are ten polynomial equations over
`Q(sqrt(2),sqrt(5),sqrt(7))`.  This is the compact algebraic root form for the
candidate; it is smaller than the older 11-variable `(p,q,r,h,theta)` radical
system because `p`, `q`, and `h` have already been eliminated.

The reproducibility script is

```text
scripts/gap3/k3_reduced_stationary_system.py
```

and the saved 100-digit solve is

```text
results/k3_reduced_stationary_system_nsolve_20260530.json
```

It gives

```text
t   = 0.79944015258441336692364609676458692162476415456096602537542826560276836937657083
r   = 0.24853630667325758439913982802189225299576325711264442763381213900445307063155603
c   = -0.89061232425680083842682268320784099344922509661131286431894043006740811130543338
s   = -0.45476333172530416857099796023518022024592865579369205974124601404424572234046462
w   = 0.00048120869234335333108797700177735285414306785300425692548301991666864883580142980
C_3 = 0.021936469459403747249299192478957700397867315103825199449732759374588683988088348
```

The 100-digit `nsolve` residual for the ten polynomial equations is about
`2.47e-88`.  This does not by itself prove global maximality, but it is now the
exact algebraic object to isolate: `C_3` is `sqrt(w_*)`, where `w_*` is the real
root of this explicit polynomial system in the displayed branch.

### Reduced Local Maximum Diagnostic

The script

```text
scripts/gap3/k3_reduced_local_check.py
```

checks the reduced three-variable objective in variables `(log t, log r,
theta)`.  The saved run

```text
results/k3_reduced_local_check_20260530_171956.json
```

found

```text
(t,r,theta) = (0.79944015077205111, 0.24853631456081229, 3.6136991174663304)
(p,q,h)     = (0.65881233598573963, 0.52543330023995172, 0.036231998883338765)
K3          = 0.0219364694594037616670600954194014775566756725311279296875
```

High-precision finite-difference checks at steps `1e-4`, `3e-5`, and `1e-5`
gave stable negative Hessian spectra:

| step | gradient inf-norm | Hessian eigenvalues |
|---:|---:|---|
| `1e-4` | `8.741e-11` | `[-0.021637875856, -0.002652874810, -0.002244999238]` |
| `3e-5` | `7.945e-11` | `[-0.021637875938, -0.002652874810, -0.002244999237]` |
| `1e-5` | `7.875e-11` | `[-0.021637875945, -0.002652874810, -0.002244999237]` |

This numerically supports a nondegenerate local maximum of the reduced kernel.
It is not a substitute for real-root isolation and boundary exclusion.

The candidate-neighborhood face optimizer

```text
scripts/gap3/k3_reduced_candidate_box.py
results/k3_reduced_candidate_box_20260530_185443.json
```

checks the boundary of the radius-`0.15` box in `(log t, log r, theta)` around
the candidate.  All six faces optimize below the center value:

| face | boundary value | drop from center |
|---|---:|---:|
| `log_t` low | `0.021695477231709521` | `2.410e-4` |
| `log_t` high | `0.021696542914793084` | `2.399e-4` |
| `log_r` low | `0.021912074016675887` | `2.440e-5` |
| `log_r` high | `0.021906297142625059` | `3.017e-5` |
| `theta` low | `0.021908502216039603` | `2.797e-5` |
| `theta` high | `0.021909605376395153` | `2.686e-5` |

Thus the tightest numerical local-box boundary margin is about `2.44e-5`.

An interval branch-and-bound prototype is now in

```text
scripts/gap3/k3_reduced_interval_branch.py
```

It evaluates interval upper bounds using a closed-form, interval-friendly
version of the eliminated direction gain.  The broad post-fix run

```text
results/k3_reduced_interval_branch_20260530_184803.json
```

covered `log t in [-6,3]`, `log r in [-6,2]`, and had no interval radicand
failures after safe nonnegative square-root enclosures were added.  The core
progress run

```text
results/k3_reduced_interval_branch_20260530_185239.json
```

covered `log t in [-2,2]`, `log r in [-4,1]`, used initial splits
`10 x 10 x 40`, and processed `30000` branch boxes.  It certified `9008` boxes
below the target `C_3 - 1e-5` and set aside `16` boxes inside the candidate
neighborhood.  The largest unresolved interval upper was still loose,
`0.024902490273661197`, on a box whose center value is only
`0.020098881239986103`.  The top unresolved center values are all below the
target by at least about `6e-4`, so the obstruction is interval overestimation,
not a discovered competing maximum.

The unresolved-box refiner

```text
scripts/gap3/k3_reduced_interval_refine.py
results/k3_reduced_interval_refine_top20_20260530.json
```

then took the 20 saved unresolved boxes from the core run and refined each one
independently.  All 20 certified below `C_3 - 1e-5`; the hardest two required
only 430 and 450 processed branch boxes.  This shows that targeted refinement
is effective, and that the remaining challenge is workflow/coverage of all
queued boxes rather than the existence of a visible competing peak.

The interval brancher is therefore useful as a proof scaffold, but not yet a
certificate.  A paper-grade interval proof will need either much sharper box
models, Taylor/mean-value bounds, or branch-local algebraic isolation.

### Reduced Global and Boundary Diagnostics

The script

```text
scripts/gap3/k3_reduced_global_scan.py
```

searches a broad compact window in `(log t, log r, theta)`, polishes the best
random samples, runs differential evolution, and separately optimizes the four
log-coordinate boundary faces.  The wide stress run

```text
results/k3_reduced_global_scan_20260530_183504.json
```

used

```text
log t in [-20,12]
log r in [-30,12]
theta in [0,2*pi]
random samples: 50000
```

and found the known interior candidate as the best point:

```text
random polished best: 0.021936469459403755
differential-evolution polished best: 0.021936469459389762
```

The best finite-window boundary face was the small-`r` face:

```text
log_t_min: 1.1368153602280092e-10
log_t_max: 0.0070146242510187046
log_r_min: 0.020469639943475681
log_r_max: 0.011594185823928509
interior minus best face: 1.466830e-3
```

The companion script

```text
scripts/gap3/k3_reduced_boundary_limits.py
```

optimizes the explicit limiting objectives on the true boundary faces.  The run

```text
results/k3_reduced_boundary_limits_20260530_183643.json
```

gave

```text
r -> 0:        0.020469639943474581
t -> infinity: 0.0070145600975660581
r -> infinity: 0.011594219375206499
best boundary limit: r -> 0
interior minus best boundary limit: 1.466830e-3
```

Together with the local Hessian check, these scans give strong numerical
evidence that the reduced-kernel candidate is the relevant interior maximum and
that the reduced boundary is safely below it.  They remain diagnostics; the
paper-ready version still needs interval or real-root isolation arguments.

## Polynomial Stationarity System

The radical-free stationarity system uses variables

```text
p, q, r, h, c, s, a, b, e, m, w
```

with

```text
c = cos(theta), s = sin(theta)
a^2 = P_+
b^2 = P_-
e^2 = Q
m = p(a+b) + sqrt(2) h r e
w = C_3^2
```

and the value relation

```text
280^2 w A^2 G = 70 q^2 m^2.
```

The stationarity equations are implemented in `scripts/gap3/k3_radical_elimination_probe.py`. This is the right starting point for an exact algebraic proof: isolate the real critical point, prove it is the relevant local maximum, and prove no other support or boundary point exceeds it.

## Scripts and Artifacts

Primary scripts:

- `scripts/gap3/k3_stationary_system.py`: builds the exact 9-mode phase-fixed system, coordinate-symmetry reductions, equivariant 8-variable system, and tiny-pair diagnostic.
- `scripts/gap3/k3_kernel_compress.py`: compresses the reduced numerator into complex monomial form and verifies the exact symbolic identity.
- `scripts/gap3/k3_three_variable_reduction.py`: verifies the analytic elimination of `p,q,h` and optimizes the resulting `(t,r,theta)` objective.
- `scripts/gap3/k3_reduced_stationary_system.py`: builds the compact 10-equation reduced algebraic system for `w=C_3^2` and solves its displayed branch at high precision.
- `scripts/gap3/k3_reduced_local_check.py`: checks the reduced objective's stationary residual and finite-difference Hessian spectrum at the candidate.
- `scripts/gap3/k3_reduced_candidate_box.py`: optimizes all faces of a small candidate neighborhood to quantify the local isolation margin.
- `scripts/gap3/k3_reduced_interval_branch.py`: interval branch-and-bound scaffold for excluding reduced-kernel boxes outside the candidate neighborhood.
- `scripts/gap3/k3_reduced_interval_refine.py`: independently refines unresolved boxes saved by the interval brancher.
- `scripts/gap3/k3_reduced_global_scan.py`: searches a broad compact reduced-kernel window and its log-coordinate boundary faces.
- `scripts/gap3/k3_reduced_boundary_limits.py`: optimizes the explicit reduced-kernel boundary limits `r -> 0`, `t -> infinity`, and `r -> infinity`.
- `scripts/gap3/k3_radical_elimination_probe.py`: builds the radical-free polynomial stationarity system and runs low-degree algebraic-dependence probes.
- `scripts/gap3/k3_sage_eliminate.sage`: Sage/Singular elimination experiments.
- `scripts/gap3/k3_sympy_modular_gb.py`: local SymPy modular Groebner fallback for the 8-variable equivariant system when Sage/Singular are unavailable.
- `scripts/gap3/k3_active_set_verify.py`: embeds the 9-mode candidate into the full k=3 DCxA nucleus, runs inactive-floor sweeps, and perturb-scans the candidate basin.
- `scripts/gap3/k3_active_hessian_check.py`: refines the 9-mode active support and checks the active-support Hessian spectrum.
- `scripts/gap3/k3_inactive_release_scan.py`: minimizes the one-sided inactive release coefficient over each inactive mode's angular/polarization variables.
- `scripts/gap3/k3_inactive_release_mpmath_check.py`: confirms worst inactive releases by high-precision objective differences.

Generated visual:

- `visuals/k3_closed_form_kernel.png`

Useful rerun commands:

```powershell
python scripts/gap3/k3_stationary_system.py
python scripts/gap3/k3_kernel_compress.py
python scripts/gap3/k3_three_variable_reduction.py
python scripts/gap3/k3_reduced_stationary_system.py --nsolve --precision 100 --save results/k3_reduced_stationary_system_nsolve_20260530.json
python scripts/gap3/k3_reduced_local_check.py
python scripts/gap3/k3_reduced_candidate_box.py
python scripts/gap3/k3_reduced_interval_branch.py --log-t-bounds=-2,2 --log-r-bounds=-4,1 --initial-splits=10,10,40 --candidate-radius=0.15,0.15,0.15 --target-margin 1e-5 --max-boxes 30000 --max-depth 36 --progress-every 5000
python scripts/gap3/k3_reduced_interval_refine.py results/k3_reduced_interval_branch_20260530_185239.json --limit 20 --max-boxes-per-input 5000 --max-depth 48 --output results/k3_reduced_interval_refine_top20_20260530.json
python scripts/gap3/k3_reduced_global_scan.py --log-t-bounds=-20,12 --log-r-bounds=-30,12 --samples 50000 --de-maxiter 320 --face-maxiter 260 --popsize 24
python scripts/gap3/k3_reduced_boundary_limits.py
python scripts/gap3/k3_radical_elimination_probe.py --skip-nsolve
python scripts/gap3/k3_sympy_modular_gb.py --summary-only
python scripts/gap3/k3_active_hessian_check.py --refine-starts 24 --refine-scales=1e-4,1e-3,1e-2,5e-2
python scripts/gap3/k3_active_set_verify.py --support-refine-starts 24 --support-refine-scales=1e-4,1e-3,1e-2,5e-2 --floors=-12,-16,-20,-24,-30,-36,-40 --basin-floor=-40
python scripts/gap3/k3_inactive_release_scan.py --active-refine-starts 24 --active-refine-scales=1e-4,1e-3,1e-2,5e-2 --release-floor=-24 --inactive-floor=-100 --samples 2048 --polish-starts 12
python scripts/gap3/k3_inactive_release_mpmath_check.py --scan-json results/k3_inactive_release_scan_20260530_refined.json --active-refine-starts 24 --active-refine-scales=1e-4,1e-3,1e-2,5e-2 --floors=-20,-24,-28,-30 --dps 140
```

Sage/Docker pattern:

```powershell
docker run --rm --entrypoint sage -v "${PWD}:/workspace" -w /workspace sagemath/sagemath:latest scripts/gap3/k3_sage_eliminate.sage
```

## Negative/Incomplete Results

Low-degree PSLQ searches did not find a small algebraic relation for `C_3`, `C_3^2`, or `1120*C_3^2` over the natural coefficient field `Q(sqrt(2),sqrt(5),sqrt(7))` within the tested bounds. Adding more radicals may still be relevant, but there is no simple small-coefficient relation yet.

Direct lexicographic elimination in Sage/Singular was too large in the current form. A grevlex computation over `GF(1039)` completed, but FGLM conversion to lex stalled. This suggests the exact proof likely needs a more targeted elimination order, real-root isolation from a smaller resultant, or a hybrid interval proof around the reduced kernel.

## Paper Integration Guidance

Paper-ready conservative integration:

1. Remove the stale equality claim `C(I_3)=0.021936432440...`.
2. State that the k=3 block is certified below `C(I_2)` and record the refined algebraic active-set candidate value `0.021936469459403...`.
3. Describe `W=C_3^2` as the isolated root of a primitive irreducible degree-40 integer polynomial, with the polynomial supplied as an ancillary certificate artifact.
4. Explain the old scan discrepancy as the `loga=-8` floor/basin miss shown by the refined full-nucleus floor sweep.

Do not state the theorem `C(I_3)=C_3` as an exact global algebraic value until the following are complete:

1. Real-root isolation for the reduced stationary system.
2. A local maximality/nondegeneracy proof for the isolated root.
3. Boundary exclusion for the reduced kernel.
4. A proof that no other k=3 support or non-equivariant perturbation exceeds the candidate.
5. A clean explanation of why the scan value and closed-form candidate differ by about `3.70e-8`.

Once those are done, the paper insertion should probably be a theorem or proposition after the current finite-block k=3 paragraph, with the scan value retained as numerical confirmation rather than the primary source of the constant.