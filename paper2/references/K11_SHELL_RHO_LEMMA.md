# k11 Shell/Rho Weighted-Row Certificate

Status: working certificate note, not paper text.

Purpose: record the finite weighted row-sum shell/fiber certificate that promotes
the k11 Route A radial shell/rho target from a conditional diagnostic to a k11
Route A closure certificate.

## Route A Inputs

The Route A paired Schur/Hilbert--Schmidt reduction gives the joint scalar
kernel

```tex
K(s,t)=M_{oa}(s,t)+M_{pmid}(s,t),
```

with endpoint

```tex
K_*={\sqrt2(110\log2+43)\over1280},
\qquad \sqrt{K_*}=0.362973375146579\ldots .
```

The existing Bernstein audit in `route_a_joint_kernel_cert.py` proves on the
coarse k11 diagnostic square `a,b in [19/20,3/2]`

```tex
|\partial_sK|\le {1\over2},
\qquad |\partial_tK|\le {3\over4},
\qquad K_{collar}\le {2\over5}.
```

With vector-cube thickening these constants give only

```text
U_11 <= 1.5104552225 > C(I_2)2^(11/2)=1.0291793439.
```

Thus cubical perturbation cannot close k11. The successful replacement is to
charge the finite Route A sum on squared-radius shell variables.

## Radial Envelope

For relay height `j`, the squared-radius mesh sizes are

```tex
\Delta s \sim h_k^2,
\qquad \Delta t,\Delta\rho \sim (2^j h_k)^2.
```

Using the same certified coarse constants gives

```text
U_11,radial <= 0.959566132239 < 1.029179343858,
margin >= 0.069613211619.
```

The direct diagnostic command is

```powershell
python scripts/gap3/route_a_joint_kernel_cert.py --skip-bernstein --skip-k12-finite --diagnose-k11-radial
```

The row certificate below supplies the missing angular/fiber domination needed
to make this radial bookkeeping rigorous.

## Weighted Row Theorem

For each represented first shell `n in [2^11,2^12)`, define the finite discrete
row

```tex
A_{11}(n)=\sum_{p\in\mathcal N_n}\sum_{q\in I_{11}}
       g(n/2^{11}, |q|^2/2^{11}, |p+q|^2/2^{11})
       \mathbf 1_{2^{11}\le |p+q|^2<2^{12}},
```

with same-shell triples deleted. Compare it to the continuum row

```tex
C_{11}(n)=\sum_{m\in I_{11}} r_3(n)r_3(m)
       {2^{11}\over 4\sqrt{nm}}K(n/2^{11},m/2^{11}).
```

The certified all-row statement is

```tex
A_{11}(n)\le R_{11} C_{11}(n)
\qquad\hbox{for every represented } n\in[2^{11},2^{12}),
```

with

```text
R_11 <= 1.011760776695.
```

This row ratio is below the squared radial budget

```text
(1.029179343858 / 0.959566132239)^2 = 1.150356100192...
```

by

```text
row-ratio margin >= 0.138595323497.
```

Therefore the final norm envelope is

```text
sqrt(R_11) * U_11,radial <= 0.965192260185
    < C(I_2)2^(11/2) = 1.029179343858,
final k11 margin >= 0.063987083673.
```

The integrated certificate command is

```powershell
python scripts/gap3/route_a_joint_kernel_cert.py --skip-bernstein --skip-k12-finite --certify-k11-weighted-row
```

The combined Route A check with the k12 finite-cell certificate is

```powershell
python scripts/gap3/route_a_joint_kernel_cert.py --skip-bernstein --certify-k11-weighted-row
```

Both commands print `PASS: k=11 weighted-row Route A envelope closes`.

## Certificate Scripts

The row computations are independent, and all row-level scripts support
row-level `--workers` threading; `--workers 0` uses all logical CPUs.

- `scripts/gap3/k11_low_shell_weighted_sweep.py` does the orbit-reduced floating
  all-row diagnostic screen.
- `scripts/gap3/k11_row_mp_verify.py` aggregates exact `(m,ell)` row counts and
  evaluates rows using high-precision or mpmath interval arithmetic.
- `scripts/gap3/k11_row_interval_screen.py` gives a vectorized outward-rounded
  interval screen.
- `scripts/gap3/k11_row_coarse_cert.py` is the proof-path all-row guard.

The all-row coarse guard command is

```powershell
python scripts/gap3/k11_row_coarse_cert.py --all --top 20 --progress-rows 200 --workers 0
```

It replaces every actual pair weight by `g_float + 1e-4` and every continuum
kernel value by `max(K_float - 1e-6,0)`, then guards positive summations by the
usual `gamma_n=n eps/(1-n eps)` factor and rounds outward with `nextafter`.
With these deliberately fat allowances, all `1707` represented k11 rows pass.
The final guarded worst rows are

```text
n=2944, r3=48,  coarse ratio <= 1.011760776695, margin 0.138595323496
n=3048, r3=144, coarse ratio <= 1.011647545299, margin 0.138708554893
n=3264, r3=48,  coarse ratio <= 1.011148077495, margin 0.139208022696
```

## Independent Interval Checks

The floating all-row screen found global worst row `n=3264` with ratio
`1.003461431741755`. The top screened rows were independently checked with
mpmath interval arithmetic:

```powershell
python scripts/gap3/k11_row_mp_verify.py --shell 3264 --shell 2944 --shell 3048 --shell 3232 --shell 3328 --dps 80 --interval
```

The interval upper ratios are

```text
n=3264: 1.003461431741755
n=2944: 1.002495942963377
n=3048: 1.00248336554164
n=3232: 1.00242328504482
n=3328: 1.002367371233262
```

Band leaders were also interval checked:

```text
r3 <= 96:        n=3264, ratio <= 1.003461431741755
97 <= r3 <=256:  n=3048, ratio <= 1.00248336554164
257<= r3 <=512:  n=3443, ratio <= 1.001931994386431
r3 >= 513:       n=3357, ratio <= 1.001626885695426
```

A supplemental direct vectorized interval run reached `1500/1707` rows before
being stopped; its running worst was already the expected `n=3264` row with
interval upper ratio `1.003461431768`. This is diagnostic support only; the
completed proof path is the coarse all-row guard plus the arithmetic audit.

## Arithmetic Audit

The vectorized formulas are written using explicit multiply chains rather than
generic integer powers. In the joint integrand,

```tex
D=4st-(\rho-s-t)^2,
```

all exact lattice inputs are dyadic and lie in `[1,2]`. Hence
`|rho-s-t|<=3`, `0<=D<=16`, and

```tex
0\le {(s-\rho)^2D^2\over 32s^{7/2}t^{7/2}\rho^2}\le 8,
\qquad
0\le {(s-t)^2D\over 8s^{3/2}t^{5/2}\rho^2}\le 2.
```

Let `u=2^{-53}` and `gamma_N=Nu/(1-Nu)`. Charging the whole nonnegative pair
weight evaluation to `N=200` rounded operations gives the crude absolute bound

```tex
\gamma_{200}\cdot 10^3 < 3\cdot 10^{-11} \ll 10^{-4}.
```

The exact shell/rho admissibility test is integer arithmetic, so the only
roundoff being guarded is formula evaluation.

For the closed-form kernel `K`, the `M_oa` numerator has absolute term sum below
`3e4` on `[1,2]^2`; the denominator is at least `320`. The `M_pmid` bracket
has absolute term sum below `17`, and its prefactor is at most `1/8`. Charging
the whole multiply-chain evaluation to `N=1000` rounded operations gives

```tex
\gamma_{1000}\cdot 3\cdot 10^4/320 < 2\cdot 10^{-11} \ll 10^{-6}.
```

Thus the script's `g_float+1e-4` and `max(K_float-1e-6,0)` replacements are far
wider than the IEEE-754 evaluation uncertainty. Positivity of the true kernel
comes from its definition as the rho-integral of the nonnegative joint
integrand, so clamping negative lower guards to zero is legitimate.

## Consequence

The k11 Route A finite-height block is closed under the weighted-row certificate:

```text
sqrt(1.011760776695) * 0.959566132239
  <= 0.965192260185
  < 1.029179343858.
```

k8-k10 are not closed by this universal Route A mechanism: even the zero-error
joint finite-height tower misses their thresholds. They require the separate
signature-specific primitive-core program described in
`references/SMALL_K_STRUCTURAL_ATLAS.md` and
`references/SPECTRAL_TAXONOMY_TAIL_REPAIR.md`.
