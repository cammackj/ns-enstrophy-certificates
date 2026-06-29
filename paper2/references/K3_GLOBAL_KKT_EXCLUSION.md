# K3 Global KKT Exclusion Theorem Draft

Status: manuscript theorem bridge, not yet inserted into `paper2/ns_cancellation.tex`.

Date: 2026-05-31

This note upgrades the existing k=3 active-set/nucleus certificate into the manuscript-facing full-block statement supported by the May 31 interval ledgers. It is written to replace the current k=3 paragraph in Proposition `prop:smallk` and to strengthen Appendix `app:k3alg` once TeX edits are approved.

## Theorem Statement

```tex
\begin{theorem}[Exact full-block maximum for $I_3$]
\label{thm:k3-fullblock-global}
Let $I_3=[8,16)$ and let
\[
  \mathcal R(v)=\frac{|B(v,v,\Delta v)|}{X(v)^2D(v)},
  \qquad X(v)=\|\nabla v\|_{L^2},\quad D(v)=\|\Delta v\|_{L^2}.
\]
Then
\[
  C(I_3)=\sup\{\mathcal R(v):v\hbox{ divergence-free and supported on }I_3\}
       = C_{3,\mathrm{alg}},
\]
where
\[
  C_{3,\mathrm{alg}}
  =0.021936469459403747249299192478957700397867315103825\ldots .
\]
Equivalently $C_{3,\mathrm{alg}}=\sqrt{W_3}$, where $W_3$ is the unique real root in the recorded isolating interval of the primitive irreducible degree-$40$ polynomial $P_3(W)\in\mathbb Z[W]$ supplied in the ancillary certificate.

In the positive-representative convention, equality is attained exactly on the signed-coordinate-permutation, conjugation, gauge, and nonzero-scale orbit of the nine-mode support
\[
\begin{gathered}
(2,2,0),\ (1,0,-3),\ (0,1,-3),\ (1,2,3),\ (2,1,3),\\
(3,1,0),\ (1,3,0),\ (2,-1,-3),\ (1,-2,3),
\end{gathered}
\]
with the algebraic coefficient ratios recorded by the active-set certificate. All other fields supported on $I_3$ satisfy
\[
  \mathcal R(v)<C_{3,\mathrm{alg}}.
\]
\end{theorem}
```

The polynomial is too large for the body text. The manuscript should cite `results/k3_w_primitive_polynomial.txt`, which records:

```text
degree: 40
height_bits: 1331
integer_factorization_over_ZZ: irreducible
real_roots_total: 14
roots_in_isolating_interval: 1
heldout_modular_matches: 3/3
```

The isolating interval for `W3` is centered at

```text
0.00048120869234335333108797700177735285414306785300425692548...
```

## Proof Draft

```tex
\begin{proof}[Proof, by finite KKT certificate]
By homogeneity, maximise $\mathcal R$ on the compact unit sphere $X(v)=1$ in the finite-dimensional real divergence-free Fourier space supported on $I_3=[8,16)$. Replacing $v$ by $-v$ if necessary fixes the numerator sign, so maximisers are KKT points for the smooth objective $R(v)=B(v,v,\Delta v)/(X^2D)$ on some coordinate face determined by their Fourier support. Thus a global maximiser is either the active interior critical point, a support-local KKT point on a proper face with nonnegative release coefficients into every omitted mode, or a point with a negative second variation of $-R$, which cannot be a local maximum.

The k=3 block contains exactly the shell values
\[
  \{8,9,10,11,12,13,14\}
\]
and, in positive representatives, $85$ Fourier modes and $5040$ triads. The certificate enumerates all $106$ triad-connected shell supports in this block. Triad-disconnected shell supports reduce to their connected components: the trilinear numerator splits across components, while the denominator is monotone under adding orthogonal components, so a maximiser may be chosen on a triad-connected shell component.

For the DCA shell triple $\{8,10,14\}$, the active-set reduction gives the algebraic stationary value $C_{3,\mathrm{alg}}=\sqrt{W_3}$ on the nine-mode support displayed above. The reduced interval certificate proves that this stationary point is the unique critical point in the certified local box and that the reduced Hessian of $R$ is strictly negative there modulo the known symmetry directions. Outside that box, the compact and tail interval ledgers for the reduced active variables give a strict upper bound below $C_{3,\mathrm{alg}}$.

It remains to exclude full-block directions transverse to the active support. The one-mode inactive numerator variation vanishes identically for every inactive mode: there is no full-block triad containing two copies of the same inactive positive mode and one active mode. The denominator variation is therefore strictly penalising in each such one-mode release. For simultaneous inactive releases, the interval inactive-Hessian ledger for $-R$ over the certified reduced local box decomposes the $304$ inactive real coordinates into $20$ components and proves every component positive definite; the worst certified lower bound is
\[
  9.42696837813\cdot 10^{-4}>0.
\]
Thus the active orbit is a strict full-block local maximiser.

The remaining shell supports are handled by the finite KKT ledger. Every shell stratum containing the DCA triple collapses, after optimisation and support extraction, to one of the $12$ positive-representative active-orbit supports. For non-DCA supports, the ledger has $33$ value clusters, reducing to $25$ canonical signed-coordinate-permutation mechanisms. All $33$ clusters have nonnegative full-block one-mode release coefficients. Among them, $20$ are support-local KKT competitors; each has an interval Krawczyk certificate for the sliced critical point, an interval positive-definiteness certificate for the Hessian of $-R$ on the slice, and an interval upper bound strictly below $C_{3,\mathrm{alg}}$. The strongest competitor is still separated by
\[
  C_{3,\mathrm{alg}}-R \ge 0.002530658096802803>0.
\]
The other $13$ non-DCA clusters have an interval-certified negative directional second derivative of $-R$ and therefore are not local maxima; the weakest such certificate has upper bound
\[
  -6.27385663456\cdot10^{-7}<0.
\]

Consequently no non-DCA KKT face can attain the active value, and every DCA-containing maximising face lies on the active orbit already isolated above. Therefore $C(I_3)=C_{3,\mathrm{alg}}$, with equality only on the stated orbit.
\end{proof}
```

## Certificate Lemmas To State Or Cite

The proof above is manuscript-ready if these finite certificate lemmas are accepted as ancillary, reproducible verification lemmas.

1. **Reduced active uniqueness.** The reduced active variables `(log_t, log_r, theta)` have a unique global maximiser at the recorded center, with value `C3`, proved by compact, tail, and local interval ledgers.

2. **Full-block local bridge.** Inactive directions around the active reduced box are strictly decreasing for `R`. One-mode inactive numerator variation is exactly zero, and the 304-dimensional simultaneous inactive Hessian of `-R` is interval-certified positive over the box.

3. **Shell-support coverage.** The k=3 block has exactly seven nonempty shell values `{8,9,10,11,12,13,14}`. All nontrivial triad-connected shell supports are the 106 subsets enumerated in the shell-stratum ledger. Disconnected shell supports reduce to connected components.

4. **DCA collapse.** Every scanned shell stratum containing `{8,10,14}` has significant support equal to one of the 12 positive-representative active orbit supports.

5. **Non-DCA KKT exclusion.** The 33 non-DCA shell-cluster mechanisms have no negative full-block one-mode release coefficients. The 20 support-local KKT mechanisms are interval-certified below target, and the 13 remaining mechanisms are interval-certified saddles.

The top-level ledger that checks these dependencies is

```text
results/k3_global_kkt_exclusion_ledger_20260531.json
```

It reports

```text
status = finite_k3_shell_cluster_kkt_exclusion_closed_at_interval_audit_level
passes = true
```

## Artifact Map

Core theorem value:

```text
results/k3_w_primitive_polynomial.txt
results/k3_reduced_stationary_system_nsolve_900.json
```

Reduced active proof:

```text
results/k3_reduced_local_certificate_r0002_20260531.json
results/k3_reduced_taylor_refine_compact_all_margin1e-5_radius020_20260531.json
results/k3_reduced_taylor_branch_candidate_r020_to_r002_margin1e-8_20260531.json
results/k3_reduced_taylor_branch_candidate_r002_to_r0002_margin1e-10_20260531.json
results/k3_reduced_taylor_branch_candidate_r0002_to_r00002_margin1e-12_20260531.json
```

Full-block active bridge:

```text
results/k3_reduced_bridge_interval_cert_r0002_dps50_parallel_20260531.json
```

Global finite KKT exclusion:

```text
results/k3_shell_strata_scan_all_max7_starts24_parallel_20260531.json
results/k3_dca_orbit_collapse_report_20260531.json
results/k3_shell_cluster_kkt_suite_all_status_params_20260531.json
results/k3_support_kkt_mp_replay_kkt_only_dps80_gap_20260531.json
results/k3_support_interval_cert_summary_kkt_only_20260531.json
results/k3_support_interval_saddle_cert_hneg_dps60_20260531.json
results/k3_global_kkt_exclusion_ledger_20260531.json
```

## Remaining Authorial Choice

There are no TeX edits in this note. Before insertion, decide whether the paper will present the result as a certified computational theorem using these ancillary interval ledgers, or whether every floating interval center and basis constant must be replaced by rational data in the artifact. The theorem/proof text above is ready for the former standard.
