# K3 Certificate Packet

Status: paper-integration audit packet, not manuscript text.

Date: 2026-05-31

## Purpose

This note records the current paper-facing proof package for the refined k=3
constant

```text
C3 = 0.021936469459403747249299192478957700397867315103825...
W3 = C3^2 = 0.00048120869234335333108797700177735285414306785300425692548...
```

The active reduced theorem is now certificate-closed: the three-variable reduced
kernel has the displayed value as its unique global maximum, up to the known
symmetry branch.  The local full-block bridge has also been interval-certified
for the inactive quadratic block over the reduced radius `2e-4` box.  The full
85-mode global equality theorem now has a finite shell-cluster KKT exclusion
ledger closed at interval-audit level; the manuscript-facing theorem bridge is
in

```text
references/K3_GLOBAL_KKT_EXCLUSION.md
```

The remaining editorial choice is whether the paper accepts these interval
ledgers as ancillary computational certificates or requires a rationalized
endpoint version of the same constants.

## Reduced Active Theorem

After the explicit p,q,h elimination, the active 9-mode k=3 kernel is a function
of

```text
log t, log r, theta.
```

The stationary point is centered at

```text
log t = -0.22384360556449978
log r = -1.3921663408045954
theta =  3.6136991131024327
```

with value

```text
0.021936469459403747249299192478957700397867315103825...
```

The high-precision algebraic stationary-system solve is saved in

```text
results/k3_reduced_stationary_system_nsolve_900.json
```

and the primitive irreducible integer polynomial for `W3=C3^2` is saved in

```text
results/k3_w_primitive_polynomial.txt
```

### Local Isolation

The candidate box was shrunk by interval Taylor certificates:

```text
results/k3_reduced_taylor_branch_candidate_r020_to_r002_margin1e-8_20260531.json
results/k3_reduced_taylor_branch_candidate_r002_to_r0002_margin1e-10_20260531.json
results/k3_reduced_taylor_branch_candidate_r0002_to_r00002_margin1e-12_20260531.json
```

All three have `unresolved_total = 0`.  The last leaves only the radius `2e-4`
box around the stationary point.

Inside that box, the local interval certificate

```text
results/k3_reduced_local_certificate_r0002_20260531.json
```

passes a Krawczyk uniqueness check for the zero of the reduced gradient and an
interval negative-definiteness check for the Hessian.  The Gershgorin row upper
bounds for the interval Hessian are

```text
-0.02019413995475618
-0.0018374489842515648
-0.0019447656273179302
```

so the reduced stationary point is the unique critical point in the residual
box and is a strict local maximum.

### Compact Middle Region

The compact middle region

```text
log t in [-2,2], log r in [-4,1], theta in [0,2*pi]
```

was certified outside the radius `0.20` candidate neighborhood by

```text
results/k3_reduced_taylor_refine_compact_all_margin1e-5_radius020_20260531.json
```

with status counts

```text
certified: 24876
candidate:   100
```

There are no unresolved boxes.  The candidate boxes are exactly the boxes later
shrunk and locally isolated above.

### Boundary And Tail Coverage

The reduced domain is compactified by direct endpoint coordinates:

```text
t       for t near 0,
tau=1/t for t near infinity,
eta=1/r for r near infinity.
```

The small-r bound is uniform for `0 <= r <= e^-4`; its script records a finite
lower log-r value only to choose the same upper radius `e^-4`.

Final tail ledgers:

| Region | Ledger | Status |
|---|---|---|
| `0 <= r <= e^-4`, `0 <= t <= e^-2` | `results/k3_reduced_rsmall_bound_t_0_em2_logr_le_m4_margin1e-5_20260531.json` | `unresolved_total=0`, top upper `0.006893200220296577` |
| `0 <= r <= e^-4`, finite middle t strip | `results/k3_reduced_rsmall_bound_logr_le_m4_logt_m100_50_margin1e-5_20260531.json` | `unresolved_total=0`, top upper `0.02190456132390771` |
| `0 <= r <= e^-4`, `t >= e^2` including `t=infinity` | `results/k3_reduced_rsmall_bound_tau_0_em2_logr_le_m4_margin1e-5_20260531.json` | `unresolved_total=0`, top upper `0.004839810624847596` |
| `e^-4 <= r <= e`, `0 <= t <= e^-2` | `results/k3_reduced_tsmall_branch_t_0_em2_logr_m4_1_margin1e-5_20260531.json` | `unresolved_total=0` |
| `e^-4 <= r <= e`, `t >= e^2` including `t=infinity` | `results/k3_reduced_tlarge_branch_tau_0_em2_logr_m4_1_margin1e-5_20260531.json` | `unresolved_total=0` |
| `r >= e` including `r=infinity`, `0 <= t <= e^-2` | `results/k3_reduced_rlarge_branch_t_0_em2_eta_0_em1_margin1e-5_20260531.json` | `unresolved_total=0` |
| `r >= e` including `r=infinity`, `t <= e^2` middle/small side | `results/k3_reduced_rlarge_branch_tle2_eta_0_em1_margin1e-5_20260531.json` | `unresolved_total=0` |
| `r >= e` and `t >= e^2`, including both endpoint faces | `results/k3_reduced_trlarge_branch_tau_0_em2_eta_0_em1_margin1e-5_20260531.json` | `unresolved_total=0` |

All tail certificates use target

```text
C3 - 1e-5 = 0.021926469459403762
```

except the local shrink certificates, which use tighter margins near the root.

Therefore, the reduced active theorem is ready to be written as a computational
interval theorem: the only equality point in the reduced variables is the
isolated stationary point above.

## Full-Block Bridge Audit

The full positive-representative k=3 block has

```text
85 modes, 5040 triads
```

under the current triad filter.  The active support has 9 modes, and its signed
coordinate-permutation orbit has 12 positive-representative supports.

The consolidated bridge ledger is

```text
results/k3_fullblock_bridge_certificate_refined_20260531.json
```

It records:

```text
active value       = 0.021936469459403255
active gradient    = 7.104e-10
active modes       = 9
inactive modes     = 76
orbit supports     = 12
```

### Exact One-Mode Release Check

For every inactive positive mode `j`, the script checks all full-block triads
and finds no triad containing exactly two copies of `j` and one active mode.
Consequently the one-inactive-mode quadratic numerator term in

```text
B(u_active + sqrt(a) v_j)
```

vanishes identically.  The one-mode release coefficient is therefore only the
positive denominator penalty.  The bridge ledger records

```text
same_inactive_active_triad_count_max = 0
```

and the floating sanity check for the corresponding 4x4 release matrices has
entries only at roundoff scale.

### Inactive Quadratic Block

The full inactive Cartesian quadratic block of `-R` has dimension `304` and
decomposes into 20 independent components.  The refined bridge ledger gives

```text
inactive Hessian min eigenvalue = +0.001537713841253...
component size counts = {1: 12, 20: 2, 28: 2, 32: 1, 52: 1, 56: 2}
```

The JSON also stores sparse upper-triangular entries for the inactive Hessian,
so the component spectra can be independently audited.

The center block has now been converted into an explicit entrywise perturbation
budget by

```text
results/k3_inactive_hessian_interval_cert_entry1e-10_20260531.json
```

For an `n` by `n` inactive component, if every Hessian entry is certified within
`+- eps`, then Weyl's inequality and `||E||_2 <= ||E||_F <= n eps` give

```text
lambda_min(A+E) >= lambda_min(A) - n eps.
```

With `eps = 1e-10`, every component passes.  The worst certified lower bound is

```text
+0.001537710641253...
```

and the largest uniform entry radius that the center certificate can tolerate is

```text
4.805355753914e-5.
```

This does not yet replace the desired high-precision interval generation of the
entries, but it fixes the target: an interval/rational entry computation with
absolute radius below `4.8e-5` is enough to certify the inactive center block.

The local active-parameter stability probe is saved in

```text
results/k3_inactive_hessian_stability_probe_36coords_componentbox_20260531.json
```

It finite-differences all 36 active polar/log-amplitude coordinates at step
`1e-5`.  The largest sampled single-coordinate sensitivity is

```text
max entry derivative    = 0.02192016...
max operator derivative = 0.02779249...
```

Using componentwise coordinate-box accounting, an active parameter sup-norm
radius `1e-4` still gives the conservative lower estimates

```text
entrywise coordinate-box lower = +0.0008965770...
operator coordinate-box lower  = +0.001510953...
```

This is a stability sizing probe, not an interval theorem.  It shows that the
remaining bridge work is plausibly small: the formal step is to replace these
center finite differences by interval derivative bounds over the reduced local
box, then feed those bounds into the same Weyl ledger.

The sharper local bridge route uses the explicit reduced-to-active map in

```text
scripts/gap3/k3_reduced_active_map.py
```

The map reconstructs the branch-correct active coefficients from
`(log t, log r, theta)`.  At the center it agrees with the refined 36-parameter
active optimizer up to the optimizer's residual/gauge noise, with maximum
coefficient difference about `2.6e-7` after the common scale is matched.

The interval enclosure for this map over the reduced local box is

```text
results/k3_reduced_active_map_interval_r0002_20260531.json
```

It confirms that the selected square-root phase branch is stable throughout the
box:

```text
v-phase branch sign = +1
unit-im interval    = [0.9926843961..., 0.9932798882...]
```

All center coefficients are contained.  The largest active coefficient interval
width is

```text
6.700163259008e-4.
```

The reduced-manifold bridge probe is

```text
results/k3_reduced_bridge_stability_r0002_20260531.json
```

It varies only the three certified reduced variables, uses finite differences at
step `1e-5`, and samples all eight corners of the reduced local box of radius
`2e-4`.  At the reduced center, the inactive Hessian of `-R` has

```text
min eigenvalue = +0.001511961420816...
```

With componentwise coordinate-box accounting over radius `2e-4`, the pessimistic
lower bounds are

```text
entrywise coordinate-box lower = +0.001366460...
operator coordinate-box lower  = +0.001504867...
```

The eight corner samples all remain positive; the smallest sampled corner
inactive eigenvalue is

```text
+0.001511192224134...
```

A smaller-step check

```text
results/k3_reduced_bridge_stability_r0002_step3e-6_20260531.json
```

reproduces the same derivative bounds to the shown precision.  This still is not
the final interval derivative theorem, but it reduces the formal local bridge to
three explicit reduced variables rather than a 36-coordinate active box.

The reduced-box interval inactive-Hessian certificate is now saved in

```text
results/k3_reduced_bridge_interval_cert_r0002_dps50_parallel_20260531.json
```

It uses the interval reduced-to-active coefficient map over the radius `2e-4`
box, enumerates the `1344` inactive quadratic triads and `1792` structural
coordinate pairs, and computes interval Hessian entries directly at `50` dps.
The off-diagonal entries were evaluated in parallel with `24` workers.

The componentwise Weyl ledger forms a midpoint matrix `M` and nonnegative radius
matrix `E_rad` for each inactive component and uses

```text
lambda_min(H) >= lambda_min(M) - min(max_i sum_j (E_rad)_ij, ||E_rad||_F).
```

All 20 inactive components pass.  The worst component has

```text
dimension                 = 32
lambda_min(midpoint)      = +0.0015119690111637055
spectral radius bound     =  0.0005692721733497221
certified lower bound     = +0.0009426968378139835
```

The largest individual Hessian-entry interval radius in this run is

```text
0.00011905484596885871.
```

This upgrades the multi-inactive local bridge from a finite-difference stability
probe to a direct interval certificate over the certified reduced local box.

This proves, at the current computational-audit level, that the active orbit is
a strict local full-block maximizer: active reduced directions are already
handled by the reduced Krawczyk/Hessian certificate, one-mode inactive releases
have exact zero numerator variation and positive denominator penalty, and the
multi-inactive quadratic block is positive definite.

## Global Full-Block Evidence And KKT Exclusion

The full-block census ledger

```text
results/k3_fullblock_census_orbit_report_20260531.json
```

used 2817 raw full-block starts over all 85 modes.  Within `1e-7` of the target,
all 259 rows are active-orbit rows directly or active-orbit rows after replay;
there are zero non-orbit near rows.  The orbit-aware penalized census found no
counterexample and has minimum sampled gap

```text
+3.565551e-14
```

This is strong global evidence and a useful guard against proof-route mistakes,
but it is not by itself a paper-grade global exclusion theorem.

The paper-grade exclusion is now supplied by the finite KKT ledger below, whose
manuscript theorem bridge is written in `references/K3_GLOBAL_KKT_EXCLUSION.md`.

The parallel finite shell-stratum inventory is saved in

```text
results/k3_shell_strata_scan_all_max7_starts24_parallel_20260531.json
```

It scans all 106 triad-connected shell subsets of the seven k=3 shells with 24
starts per stratum and 24 process workers.  The theorem-relevant summary is:

```text
shell strata scanned              = 106
target-level rows within 1e-9     = 16
target-level rows all contain DCA = true
DCA rows all collapse to 9 modes  = true
best non-DCA shells               = {8,9,11,13}
best non-DCA value                = 0.019405811362467...
best non-DCA gap to C3            = +0.002530658096936...
```

The DCA collapse statement is saved explicitly in

```text
results/k3_dca_orbit_collapse_report_20260531.json
```

At X-energy cutoff `1e-6`, all 16 DCA-containing shell rows have significant
support size 9 and all 16 supports lie in the 12-element signed-coordinate-
permutation orbit of the active support.

Thus, at the shell-support level, every target-level stratum is a stratum
containing the active `{8,10,14}` shell triple and the optimizer collapses back
to the same 9 active modes.  The strongest observed non-DCA competitor is the
`{8,9,11,13}` family, still separated from `C3` by about `2.53e-3`.

The best non-DCA stratum has been promoted from a scan row to a local KKT brick:

```text
results/k3_shell_stratum_local_cert_best_non_dca_20260531.json
results/k3_shell_stratum_local_cert_best_non_dca_step3e-5_20260531.json
```

For shells `{8,9,11,13}`, the 60-dps value is

```text
0.0194058113624665857148978925488149347752636570918336160400207
```

with gap

```text
C3 - value = +0.002530658096937158...
```

The projected gradient is `5.263e-10`.  The parallel Hessian check for `-R`
has, at both finite-difference steps `1e-5` and `3e-5`,

```text
0 negative / 171 flat / 9 positive
```

above tolerance `1e-7`; the largest positive eigenvalue is about `0.04088087`.
This certifies the observed best non-DCA row as a local maximum/minimum of the
correct sign for `R`, with a large separation from the target.  It is not yet a
global certificate for all non-DCA rows, but it is the first finite KKT brick
for the shell-support exclusion route.

The second non-DCA value cluster, represented by shells `{8,13}`, is saved in

```text
results/k3_shell_stratum_local_cert_non_dca_cluster1_20260531.json
```

It has value `0.018363486154938...`, gap `+0.003572983304466...`, projected
gradient `2.127e-9`, and Hessian counts for `-R` equal to
`0 negative / 58 flat / 14 positive` at step `1e-5`.

The non-DCA shell-stratum evidence has now been promoted to an all-cluster
finite KKT ledger:

```text
results/k3_shell_cluster_kkt_suite_all_status_params_20260531.json
results/k3_support_mechanism_inventory_status_params_20260531.json
results/k3_support_kkt_mp_replay_kkt_only_dps80_gap_20260531.json
```

The suite clusters the 90 non-DCA shell rows into 33 value clusters, extracts
the effective X-energy support of each representative, repolishes that exact
support with `X2=1` normalization, computes the support Hessian of `-R`, and
checks one-mode full-block release eigenvalues for every omitted positive mode.

The resulting finite ledger is:

```text
non-DCA value clusters                  = 33
canonical support mechanisms             = 25
negative full-block one-mode releases    = 0
support-local KKT competitors            = 20
strict non-KKT saddle rows               = 13
best support-local KKT competitor        = 0.019405811362464784763161859832831171908576955908248656810137205170036187609662987
best support-local KKT gap to C3         = +0.0025306580969389624861373326461265284892903591955763431898627948299638123903370129
best support-local KKT min release coeff = +0.43659085017586297
```

The 13 saddle rows remain safely below target and have explicit negative
Hessian directions for `-R`, so they are not KKT competitors.  Lowering the
effective-support cutoff to `1e-10` did not remove those negative directions;
the rerun is saved in

```text
results/k3_shell_cluster_kkt_suite_hneg_cut1e-10_20260531.json
```

Thus the current finite picture is sharper than the shell scan: every observed
non-DCA support mechanism is either a support-local KKT competitor with a large
gap and positive full-block release coefficients, or is not a KKT point at all.
The remaining formal step is to interval/rationalize this finite KKT ledger and
write the KKT-enumeration/coverage argument; the ledger itself is now replayable
because it stores the polished normalized support parameters.

That interval audit layer is now present in

```text
results/k3_support_interval_cert_summary_kkt_only_20260531.json
results/k3_support_interval_saddle_cert_hneg_dps60_20260531.json
results/k3_global_kkt_exclusion_ledger_20260531.json
```

For the 20 support-local non-DCA KKT competitors, the interval local certificate
uses second-order interval automatic differentiation on a QR/exhaustive selected
coordinate slice, Krawczyk uniqueness for the sliced critical point, and an
interval positive-definiteness check for the Hessian of `-R`.  All 20 pass.  The
worst certified KKT gap lower bound is still the best non-DCA mechanism:

```text
+0.002530658096802803
```

For the 13 Hessian-negative non-DCA rows, a one-dimensional interval AD check
certifies an explicit negative second derivative of `-R` in the stored negative
eigenvector direction.  All 13 pass; the weakest certified upper bound is still
negative:

```text
-6.273856634568238e-7
```

The assembled ledger reports

```text
status = finite_k3_shell_cluster_kkt_exclusion_closed_at_interval_audit_level
passes = true
```

This closes the finite shell-cluster KKT audit ledger: DCA rows collapse to the
active orbit, non-DCA KKT rows are interval-certified below target, and non-DCA
saddle rows are interval-certified as not local maxima.  A final manuscript
theorem still needs the prose coverage argument translating this finite ledger
into exhaustive KKT enumeration, plus exact/rational replacement of floating
basis constants if that is required by the final proof standard.

A broad active/complement flattening majorant was also tested as a possible
one-shot global exclusion route:

```text
results/k3_active_complement_majorant_probe1_20260531.json
```

For one active-orbit support it gives the much-too-large upper bound

```text
0.306870790908442
```

with the complement-only tensor dominating.  This confirms that unstructured
Schur/flattening control is not the right proof mechanism for the full equality
theorem; the remaining global proof has to use the finite KKT/shell-support
structure rather than a coarse active/complement norm majorant.

## Paper-Ready Statements

### Safe Conservative Statement

The following is safe for paper integration after prose review:

```text
The k=3 active reduced kernel has a certified unique global maximum
C3 = 0.021936469459403747... < C(I_2), and the full-block computations are
consistent with the assertion that the complete k=3 block maximum is attained
only on the signed-coordinate-permutation orbit of this active field.
For any theorem needing only the strict comparison with k=2, we may use the
conservative certified numerical upper bound C(I_3) <= 0.021936470.
```

### Equality Theorem Gate

The stronger statement

```text
C(I_3) = C3, with equality only on the active orbit
```

still needs one of the following before being presented as a fully formal paper
theorem:

1. A finite KKT/support argument excluding all non-active boundary faces and
   all non-DCA shell strata.  The local inactive 304-dimensional bridge over
   the reduced box is now interval-certified.  The finite shell-cluster KKT
   audit ledger outside those active-orbit neighborhoods is also interval-closed
   in `results/k3_global_kkt_exclusion_ledger_20260531.json`; the remaining
   paper step is the written exhaustive-coverage argument and rationalization
   level chosen for the proof.
2. A global orbit-aware penalized inequality
   `R(u) + mu * dist_orbit(u) <= C3` with a finite certificate.
3. A complete algebraic KKT/support enumeration proving every full-block
   maximizer has active support in the 12-support orbit.

The new bridge certificate and shell inventory make option 1 the shortest
route: the local block has a certified positive lower bound over the reduced
radius-`2e-4` box; the one-mode release term has an exact combinatorial zero;
all observed DCA shell strata collapse to the active 9-mode support; and all
observed non-DCA shell strata have a visible `2.5e-3` gap.  The remaining work
is to replace the shell-stratum optimizer evidence by finite interval/KKT
certificates.

## Reproduction Commands

```powershell
python scripts/gap3/k3_reduced_local_certificate.py --radius=0.0002,0.0002,0.0002 --output results/k3_reduced_local_certificate_r0002_20260531.json
python scripts/gap3/k3_fullblock_bridge_certificate.py --active-refine-starts 24 --active-refine-scales=1e-4,1e-3,1e-2,5e-2 --include-matrix --output results/k3_fullblock_bridge_certificate_refined_20260531.json
python scripts/gap3/k3_inactive_hessian_interval_cert.py --bridge-json results/k3_fullblock_bridge_certificate_refined_20260531.json --entry-radius 1e-10 --output results/k3_inactive_hessian_interval_cert_entry1e-10_20260531.json
python scripts/gap3/k3_inactive_hessian_stability_probe.py --direction-count 36 --step 1e-5 --test-radii=1e-6,3e-6,1e-5,3e-5,1e-4 --output results/k3_inactive_hessian_stability_probe_36coords_componentbox_20260531.json
python scripts/gap3/k3_reduced_active_map_interval.py --radius=0.0002,0.0002,0.0002 --output results/k3_reduced_active_map_interval_r0002_20260531.json
python scripts/gap3/k3_reduced_bridge_stability.py --radius=0.0002,0.0002,0.0002 --step 1e-5 --test-radii=1e-5,3e-5,1e-4,2e-4 --output results/k3_reduced_bridge_stability_r0002_20260531.json
python scripts/gap3/k3_reduced_bridge_stability.py --radius=0.0002,0.0002,0.0002 --step 3e-6 --test-radii=2e-4 --skip-corners --output results/k3_reduced_bridge_stability_r0002_step3e-6_20260531.json
python scripts/gap3/k3_reduced_bridge_interval_cert.py --radius=0.0002,0.0002,0.0002 --dps 50 --workers 0 --chunksize 16 --progress-every 200 --output results/k3_reduced_bridge_interval_cert_r0002_dps50_parallel_20260531.json
python scripts/gap3/k3_shell_strata_scan.py --max-size 7 --starts 24 --maxiter 2000 --workers 0 --save-params --output results/k3_shell_strata_scan_all_max7_starts24_parallel_20260531.json
python scripts/gap3/k3_dca_orbit_collapse_report.py --strata-json results/k3_shell_strata_scan_all_max7_starts24_parallel_20260531.json --cutoff 1e-6 --output results/k3_dca_orbit_collapse_report_20260531.json
python scripts/gap3/k3_active_complement_majorant.py --support-limit 1 --workers 1 --output results/k3_active_complement_majorant_probe1_20260531.json
python scripts/gap3/k3_shell_stratum_local_cert.py --strata-json results/k3_shell_strata_scan_all_max7_starts24_parallel_20260531.json --require-non-dca --rank 0 --dps 60 --hessian-step 1e-5 --workers 0 --output results/k3_shell_stratum_local_cert_best_non_dca_20260531.json
python scripts/gap3/k3_shell_stratum_local_cert.py --strata-json results/k3_shell_strata_scan_all_max7_starts24_parallel_20260531.json --require-non-dca --rank 0 --dps 60 --hessian-step 3e-5 --workers 0 --output results/k3_shell_stratum_local_cert_best_non_dca_step3e-5_20260531.json
python scripts/gap3/k3_shell_stratum_local_cert.py --strata-json results/k3_shell_strata_scan_all_max7_starts24_parallel_20260531.json --require-non-dca --rank 12 --dps 60 --hessian-step 1e-5 --workers 0 --output results/k3_shell_stratum_local_cert_non_dca_cluster1_20260531.json
python scripts/gap3/k3_shell_cluster_kkt_suite.py --strata-json results/k3_shell_strata_scan_all_max7_starts24_parallel_20260531.json --workers 0 --maxiter 3000 --output results/k3_shell_cluster_kkt_suite_all_status_params_20260531.json
python scripts/gap3/k3_shell_cluster_kkt_suite.py --strata-json results/k3_shell_strata_scan_all_max7_starts24_parallel_20260531.json --clusters 2,3,4,5,6,11,12,13,14,15,16,21,22 --support-cutoff 1e-10 --workers 0 --maxiter 5000 --output results/k3_shell_cluster_kkt_suite_hneg_cut1e-10_20260531.json
python scripts/gap3/k3_support_mechanism_inventory.py --kkt-json results/k3_shell_cluster_kkt_suite_all_status_params_20260531.json --output results/k3_support_mechanism_inventory_status_params_20260531.json
python scripts/gap3/k3_support_kkt_mp_replay.py --kkt-json results/k3_shell_cluster_kkt_suite_all_status_params_20260531.json --kkt-only --dps 80 --workers 0 --output results/k3_support_kkt_mp_replay_kkt_only_dps80_gap_20260531.json
python scripts/gap3/k3_support_interval_cert_batch.py --kkt-json results/k3_shell_cluster_kkt_suite_all_status_params_20260531.json --kkt-only --radius 1e-5 --dps 60 --workers 0 --output-dir results --summary results/k3_support_interval_cert_batch_kkt_only_r1e-5_dps60_20260531.json
python scripts/gap3/k3_support_interval_local_cert.py --kkt-json results/k3_shell_cluster_kkt_suite_all_status_params_20260531.json --cluster 9 --radius 3e-5 --dps 60 --slice-polish --output results/k3_support_interval_local_cert_cluster9_slice_polish_r3e-5_dps60_20260531.json
python scripts/gap3/k3_support_interval_local_cert.py --kkt-json results/k3_shell_cluster_kkt_suite_all_status_params_20260531.json --cluster 17 --radius 5e-6 --dps 60 --slice-polish --output results/k3_support_interval_local_cert_cluster17_slice_polish_r5e-6_dps60_20260531.json
python scripts/gap3/k3_support_interval_local_cert.py --kkt-json results/k3_shell_cluster_kkt_suite_all_status_params_20260531.json --cluster 20 --radius 1e-6 --dps 60 --slice-polish --output results/k3_support_interval_local_cert_cluster20_root_polish_r1e-6_dps60_20260531.json
python scripts/gap3/k3_support_interval_local_cert.py --kkt-json results/k3_shell_cluster_kkt_suite_all_status_params_20260531.json --cluster 26 --radius 3e-7 --dps 60 --slice-polish --output results/k3_support_interval_local_cert_cluster26_root_polish_r3e-7_dps60_20260531.json
python scripts/gap3/k3_support_interval_local_cert.py --kkt-json results/k3_shell_cluster_kkt_suite_all_status_params_20260531.json --cluster 27 --radius 1e-6 --dps 60 --slice-polish --output results/k3_support_interval_local_cert_cluster27_root_polish_r1e-6_dps60_20260531.json
python scripts/gap3/k3_support_interval_local_cert.py --kkt-json results/k3_shell_cluster_kkt_suite_all_status_params_20260531.json --cluster 29 --radius 1e-6 --dps 60 --slice-polish --output results/k3_support_interval_local_cert_cluster29_root_polish_r1e-6_dps60_20260531.json
python scripts/gap3/k3_support_interval_local_cert.py --kkt-json results/k3_shell_cluster_kkt_suite_all_status_params_20260531.json --cluster 30 --radius 1e-6 --dps 60 --slice-polish --output results/k3_support_interval_local_cert_cluster30_root_polish_r1e-6_dps60_20260531.json
python scripts/gap3/k3_support_interval_cert_summary.py --kkt-json results/k3_shell_cluster_kkt_suite_all_status_params_20260531.json --output results/k3_support_interval_cert_summary_kkt_only_20260531.json
python scripts/gap3/k3_support_interval_saddle_cert.py --kkt-json results/k3_shell_cluster_kkt_suite_all_status_params_20260531.json --dps 60 --workers 0 --output results/k3_support_interval_saddle_cert_hneg_dps60_20260531.json
python scripts/gap3/k3_global_kkt_exclusion_ledger.py --strata-json results/k3_shell_strata_scan_all_max7_starts24_parallel_20260531.json --dca-report results/k3_dca_orbit_collapse_report_20260531.json --kkt-suite results/k3_shell_cluster_kkt_suite_all_status_params_20260531.json --kkt-interval-summary results/k3_support_interval_cert_summary_kkt_only_20260531.json --saddle-cert results/k3_support_interval_saddle_cert_hneg_dps60_20260531.json --output results/k3_global_kkt_exclusion_ledger_20260531.json
```

The final reduced tail ledgers are generated by:

```powershell
python scripts/gap3/k3_reduced_rsmall_bound.py --t-bounds=0,0.1353352832366127 --log-r-bounds=-100,-4 --initial-splits 128 --target-margin 1e-5 --output results/k3_reduced_rsmall_bound_t_0_em2_logr_le_m4_margin1e-5_20260531.json
python scripts/gap3/k3_reduced_rsmall_bound.py --log-t-bounds=-100,50 --log-r-bounds=-100,-4 --initial-splits 768 --target-margin 1e-5 --output results/k3_reduced_rsmall_bound_logr_le_m4_logt_m100_50_margin1e-5_20260531.json
python scripts/gap3/k3_reduced_rsmall_bound.py --tau-bounds=0,0.1353352832366127 --log-r-bounds=-100,-4 --initial-splits 128 --target-margin 1e-5 --output results/k3_reduced_rsmall_bound_tau_0_em2_logr_le_m4_margin1e-5_20260531.json
python scripts/gap3/k3_reduced_tsmall_branch.py --t-bounds=0,0.1353352832366127 --log-r-bounds=-4,1 --theta-bounds=0,6.283185307179586 --initial-splits=8,10,40 --target-margin 1e-5 --output results/k3_reduced_tsmall_branch_t_0_em2_logr_m4_1_margin1e-5_20260531.json
python scripts/gap3/k3_reduced_tlarge_branch.py --tau-bounds=0,0.1353352832366127 --log-r-bounds=-4,1 --theta-bounds=0,6.283185307179586 --initial-splits=8,10,40 --target-margin 1e-5 --output results/k3_reduced_tlarge_branch_tau_0_em2_logr_m4_1_margin1e-5_20260531.json
python scripts/gap3/k3_reduced_rlarge_branch.py --t-bounds=0,0.1353352832366127 --eta-bounds=0,0.36787944117144233 --theta-bounds=0,6.283185307179586 --initial-splits=8,8,48 --target-margin 1e-5 --output results/k3_reduced_rlarge_branch_t_0_em2_eta_0_em1_margin1e-5_20260531.json
python scripts/gap3/k3_reduced_rlarge_branch.py --log-t-bounds=-20,2 --eta-bounds=0,0.36787944117144233 --theta-bounds=0,6.283185307179586 --initial-splits=12,8,48 --target-margin 1e-5 --output results/k3_reduced_rlarge_branch_tle2_eta_0_em1_margin1e-5_20260531.json
python scripts/gap3/k3_reduced_trlarge_branch.py --tau-bounds=0,0.1353352832366127 --eta-bounds=0,0.36787944117144233 --theta-bounds=0,6.283185307179586 --initial-splits=8,8,48 --target-margin 1e-5 --output results/k3_reduced_trlarge_branch_tau_0_em2_eta_0_em1_margin1e-5_20260531.json
```