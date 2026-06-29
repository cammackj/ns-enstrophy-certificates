# Paper Theorem Ledger and Dependency Map

Working companion document, not paper text.

Purpose: turn the current certificate package into a clean paper-facing theorem
architecture.  This document avoids bracket-display TeX; formulas are written
inline or in fenced blocks for readability.

The next-stage theorem/proof skeletons live in
`references/PAPER_THEOREM_DRAFTS.md`.

The manuscript staleness audit lives in
`references/PAPER_STALENESS_AUDIT.md`.

Candidate manuscript replacement prose lives in
`references/PAPER_MANUSCRIPT_REWRITE_DRAFTS.md`.

The pre-edit proof review checklist lives in
`references/PAPER_PROOF_REVIEW_CHECKLIST.md`.

The focused mathematical review lives in
`references/PAPER_MATHEMATICAL_REVIEW.md`.

## Naming Rule

Do not use internal A/B route labels in paper-facing prose.  Use descriptive
mathematical names:

| Paper-facing name | Covers | Working role |
|-------------------|--------|--------------|
| finite low-block certificate ledger | k=0--7 | exact, mpmath, Hessian, and GPU finite certificates |
| midpoint-centered finite-block estimate | k=8--10 | centered kernel plus row guards and low-block splice |
| joint Schur finite-cell estimate | k=12 | Schur-level paired-kernel finite-cell certificate |
| weighted-row joint Schur estimate | k=11 | row-weighted finite replacement for the k=11 endpoint |
| effective tail estimate | k>=13 | finite-k tail certificate from the taxonomy theorem |
| dyadic shell comparison theorem | all k in scope | final assembly against the exact k=2 value |

Historical script filenames may still contain older working names.  Those names
are audit-path identifiers only; they should not become theorem names.

## Target Statement

Paper target, in words:

```text
For every dyadic shell block in the paper's range except k=2,
C(I_k) < C(I_2), where C(I_2)=0.022741865409341... is exact.
```

Equivalent rescaled comparison:

```text
C_res(k) < C(I_2) * 2^(k/2), where C_res(k)=2^(k/2) C(I_k).
```

The proof should not claim exact optimizer values for k=8, k=9, or k=10.  Those
blocks are closed by certified upper bounds.

## Theorem Ledger

| ID | Paper-facing theorem or lemma | Statement needed in the paper | Proof inputs | Audit evidence |
|----|-------------------------------|-------------------------------|--------------|----------------|
| L1 | finite low-block certificate ledger | k=0, k=1, and k=3--7 are strictly below C(I_2); k=2 is exact reference | existing exact/numerical finite certificates | existing paper values and finite cert logs |
| L2 | midpoint-centered kernel bound | centered scalar kernel satisfies K_c<=1/60, derivative bounds <=1/25, collar<=9/128 | B(u,u,u)=0, midpoint output shift, Bernstein positivity | `python scripts/gap3/route_b_centered_kernel_cert.py` |
| L3 | centered row-guard lemma | centered row discretization guards hold for reduced blocks needed in k=8--10 | interval row verification and guarded coarse row screens | `centered_row_mp_verify.py`, `centered_row_coarse_cert.py` |
| T1 | midpoint-centered finite-block estimate | C(I_8), C(I_9), C(I_10) are all strictly below C(I_2) | L2, L3, certified k=1--7 splice constants | `route_b_centered_kernel_cert.py` mixed budget |
| L4 | joint Schur endpoint lemma | paired-kernel Schur endpoint bound and finite-cell k=12 estimate | Hilbert--Schmidt mixed trace sign, Bernstein endpoint certificate | `python scripts/gap3/route_a_joint_kernel_cert.py` |
| L5 | weighted-row joint Schur estimate | k=11 is strictly below C(I_2) after row inflation guard | k=11 radial envelope and row domination certificate | `python scripts/gap3/route_a_joint_kernel_cert.py --skip-bernstein --certify-k11-weighted-row` |
| T2 | joint Schur finite-block estimate | k=11 and k=12 are strictly below C(I_2) | L4 and L5 | same joint Schur audit script |
| T3 | effective tail estimate | every k>=13 is strictly below C(I_2) | primitive-core/relay continuum majorants and finite-k perturbation | `python scripts/gap3/tail_cert.py` |
| T4 | dyadic shell comparison theorem | all blocks in scope except k=2 are strictly below C(I_2) | L1, T1, T2, T3 | theorem assembly table below |

## Final Proof Chain

Use the exact k=2 value as the comparison constant:

```text
C(I_2)=0.022741865409341...
```

Then cover the shells by disjoint ranges.

| Range | Bound or value | Strict comparison |
|-------|----------------|-------------------|
| k=0 | C(I_0)=0 | below C(I_2) |
| k=1 | C(I_1)=0 | below C(I_2) |
| k=2 | C(I_2)=0.022741865409341... | exact comparison block |
| k=3 | C(I_3)<=0.021936470 | below C(I_2); refined algebraic/nucleus comparison certificate |
| k=4 | C(I_4)=0.021064396605547... | below C(I_2) |
| k=5 | C(I_5)=0.018479317637642... | below C(I_2) |
| k=6 | C(I_6)=0.020443793141444... | below C(I_2) |
| k=7 | C(I_7)=0.020280600900469... | below C(I_2) |
| k=8 | C(I_8)<=0.021262658939 | below C(I_2) |
| k=9 | C(I_9)<=0.016346812706 | below C(I_2) |
| k=10 | C(I_10)<=0.011701241970 | below C(I_2) |
| k=11 | C(I_11)<=0.021327937261 | below C(I_2) |
| k=12 | C(I_12)<=0.020366626807 | below C(I_2) |
| k>=13 | C_res(k)<=U(k)<C(I_2)2^(k/2) | below C(I_2) |

Strictness for k=8--12 follows from the chain:

```text
C(I_k) <= certified_upper(k) < C(I_2).
```

Strictness for k>=13 follows from the rescaled chain:

```text
C_res(k) <= U(k) < C(I_2) * 2^(k/2), hence C(I_k) < C(I_2).
```

Thus the final assembly proves the target statement for every block in the
paper's range except the exact comparison block k=2.

## Certified Constants to Carry Forward

### Midpoint-Centered Finite-Block Estimate

Kernel constants:

```text
K_c <= 1/60
|partial_s K_c| <= 1/25
|partial_t K_c| <= 1/25
collar <= 9/128
```

Row guards used by the mixed budget:

```text
R_6, R_8 <= 5/4
R_7, R_9, R_10 <= 3/2
```

Worst row evidence:

```text
k6  R_upper=1.103121376559912  at n=88    interval
k8  R_upper=1.028818401493566  at n=438   interval
k7  R_upper=1.037908796942949  at n=163   interval
k9  R_upper=1.007609447255     at n=512   guarded coarse
k10 R_upper=1.004985996223     at n=1968  guarded coarse
```

Mixed rescaled budgets:

```text
k=8   U_mixed <= 0.340202543023 < 0.363869846549
k=9   U_mixed <= 0.369886147671 < 0.514589671929
k=10  U_mixed <= 0.374439743029 < 0.727739693099
```

Unscaled consequences:

```text
C(I_8)  <= 0.021262658939
C(I_9)  <= 0.016346812706
C(I_10) <= 0.011701241970
```

### Joint Schur and Weighted-Row Finite Estimates

k=12 finite-cell certificate:

```text
U_12 <= 1.303464115585 < C(I_2) * 2^6 = 1.455479386198
C(I_12) <= 0.020366626807
```

k=11 weighted-row certificate:

```text
R_11 <= 1.011760776695
sqrt(R_11) * U_11,radial <= 0.965192260185
C(I_2) * 2^(11/2) = 1.029179343858
C(I_11) <= 0.021327937261
```

### Effective Tail Estimate

First tail block:

```text
U(13) < 1.910577984
C(I_2) * 2^(13/2) = 2.0583586877...
margin > 0.14778070
```

For k>13, the mesh term decreases and the comparison threshold increases, so the
k=13 certificate propagates to every k>=13.

## Audit Appendix Draft

Run from the repository root.

Tail certificate:

```powershell
python scripts/gap3/tail_cert.py
```

Joint Schur finite-cell and k=12 certificate:

```powershell
python scripts/gap3/route_a_joint_kernel_cert.py
```

k=11 weighted-row joint Schur certificate:

```powershell
python scripts/gap3/route_a_joint_kernel_cert.py --skip-bernstein --certify-k11-weighted-row
```

Midpoint-centered kernel and mixed k=8--10 budget:

```powershell
python scripts/gap3/route_b_centered_kernel_cert.py
```

Interval row guards used by the midpoint-centered proof:

```powershell
python scripts/gap3/centered_row_mp_verify.py --k 8 --all --interval --budget-ratio 1.25 --workers 16 --top 8
python scripts/gap3/centered_row_mp_verify.py --k 6 --all --interval --budget-ratio 1.25 --workers 16 --top 8
python scripts/gap3/centered_row_mp_verify.py --k 7 --all --interval --budget-ratio 1.5  --workers 16 --top 8
```

Guarded coarse row screens for the wide-margin top rows:

```powershell
python scripts/gap3/centered_row_coarse_cert.py --k 9 --all --workers 16 --progress-rows 64 --top 8 --budget-ratio 1.5 --eps-g 1e-8 --eps-k 1e-10
python scripts/gap3/centered_row_coarse_cert.py --k 10 --all --workers 16 --progress-rows 64 --top 8 --budget-ratio 1.5 --eps-g 1e-8 --eps-k 1e-10
```

## Non-Inputs and Warnings

Do not use these as proof inputs:

| Item | Why not |
|------|---------|
| k=8 Adam incumbent near 0.020628554 | useful diagnostic only; not a certified lower bound or global value |
| stale k=8, k=9, k=10 restricted-run values | restricted scans, not full-block values |
| exact optimizer values for k=8--10 | not known and not needed |
| internal A/B route labels | working labels only, not paper terminology |

## Manuscript Integration Gate

Before editing a paper `.tex` file, write the exact paper theorem statements and
proof paragraphs from this ledger, check them against the scripts above, and get
explicit approval to edit the manuscript.