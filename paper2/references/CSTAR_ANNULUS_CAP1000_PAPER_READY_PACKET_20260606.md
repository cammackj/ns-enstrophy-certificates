# C* Annulus Cap-1000 Paper-Ready Packet

This is the active proof packet for the `C^*` annulus value.  It supersedes
the earlier `0.3122642497...` full-split packet and should be the source of
truth for manuscript insertion after author approval.

## Verdict

The broad numerical search phase is done.  The cap-released finite
shell-split certificate is paper-ready as a computer-assisted finite-face
theorem with displayed target

```text
C^*_{ann,finite} <= 0.31226427.
```

The certified candidate value is

```text
0.31226426322102058.
```

The difference between the candidate and the display theorem target is

```text
6.778979433352816e-09.
```

This is excellent: the old artificial `scale_max=200` obstruction is gone, the
cap-1000 polish gave no meaningful gain, and the Hessian/KKT certificate has
now been recomputed on the correct released free face.

Important scope line: this packet is paper-ready for the complete
exactly-one-high annulus shell-split theorem.  The original base cutoff is
`N=565`, so exactly-one-high support ends at `4N=2260`, which is precisely the
endpoint of the certified finite table.  A global unrestricted `C^*` upper
statement would require a separate full-field or multi-high tail theorem; that
stronger theorem is not a dependency of the one-high annulus certificate.

## Minimal Artifact Set

Only these artifacts are needed for the paper-facing certificate.  The older
prefix scans and exploratory windows should not be cited.

| role | artifact |
|---|---|
| final candidate | `scripts/results/cstar_annulus_fullsplit_cap1000_polish_20260606.json` |
| direct Hessian on released face | `scripts/results/cstar_annulus_direct_hessian_fullsplit_cap1000_free_20260606.json` |
| theorem-gate ledger | `scripts/results/cstar_annulus_fullsplit_theorem_gate_verification_cap1000_20260606.json` |
| readable gate packet | `references/CSTAR_ANNULUS_FULLSPLIT_THEOREM_GATE_VERIFICATION_CAP1000_20260606.md` |
| TeX insert draft | `references/CSTAR_ANNULUS_CAP1000_MANUSCRIPT_INSERT_20260606.tex` |

## Recommended Evidence Model

Use a hybrid proof.

The paper should contain the theorem statement, the KKT/Hessian/cutoff-zero
logic, the rounding lemma, and the far-tail interface.  It should not contain a
large dump of JSON or exploratory tables.  The JSON/Hessian files should be
archived as a small reproducibility bundle, preferably on OSF with SHA256
hashes.

An artifact-free analytic proof is possible in principle, but it is not the
recommended route for this paper version.  It would require replacing the
`647`-variable released-face KKT/Hessian certificate by a structural theorem
that explains the active set, stationarity, and Schur negativity from shell
geometry alone.  That is a substantial new project.  The current optimizer did
not collapse to a small \(k=2\)-style algebraic atom; it is a large finite
annulus object with a clean certificate.

Recommended manuscript wording:

```text
The finite certificate is computer-assisted.  The proof uses only the archived
candidate, released-face Hessian, theorem-gate ledger, and verification scripts;
all exploratory scans are excluded from the certificate.
```

## Certificate Bundle Manifest

The minimal OSF/supplement bundle should contain the files below.  The hash
values were computed with SHA256 on 2026-06-06.

| role | file | SHA256 |
|---|---|---|
| proof packet | `references/CSTAR_ANNULUS_CAP1000_PAPER_READY_PACKET_20260606.md` | compute on archived copy |
| manuscript insert | `references/CSTAR_ANNULUS_CAP1000_MANUSCRIPT_INSERT_20260606.tex` | `D9C26A8AB9FC7F75AFA252786EB2557E187E27D1644591508FCFEF3C1A56F839` |
| compiled insert | `references/CSTAR_ANNULUS_CAP1000_MANUSCRIPT_INSERT_20260606.pdf` | `A399EB8DF0BFCF5C2F5276F6C80FBCF793269A92EBA1BBF3C184B4CC60609664` |
| readable theorem gate | `references/CSTAR_ANNULUS_FULLSPLIT_THEOREM_GATE_VERIFICATION_CAP1000_20260606.md` | `AC97A05A978F49D0F3253B878FE41651C3EC06999DADD58F6BC757992C6CE519` |
| final candidate JSON | `scripts/results/cstar_annulus_fullsplit_cap1000_polish_20260606.json` | `4873382B2A8A42F7BD2606E96A682FC7AC259DB358646FABCAC6C0FCF7B5E01E` |
| released-face Hessian JSON | `scripts/results/cstar_annulus_direct_hessian_fullsplit_cap1000_free_20260606.json` | `E0E436341A4BCBE68D190DA14FBA5314412CA06DA6F6EDBF786EFA2C00C2EAC3` |
| theorem-gate JSON | `scripts/results/cstar_annulus_fullsplit_theorem_gate_verification_cap1000_20260606.json` | `FAC6436C0841BFE68C2E44B9D83AC3058F73421B6914F497A488A8091BCFB812` |
| theorem-gate verifier | `scripts/certify_cstar_fullsplit_stabilized_gate.py` | `A1591E6627D00933B003CDE3FE5B07909A6AD3A2A53B26975324B294B71C9737` |
| direct FFT/Hessian tool | `scripts/analyze_cstar_annulus_block_schur_gate.py` | `54B18D9BF8E05FFF79870DC8E5D7EDE7A94F5FF6A96E0EBB8A3CA9532E8F4269` |

The current repository commit at the time of packaging was:

```text
38169aebbc669cf7a41e78050e976db399bc8535
```

Several files above are untracked or gitignored in the working tree, so the
hashes are the authoritative archive identifiers until the bundle is committed
or uploaded.

### Verification Commands

The finite theorem-gate ledger is regenerated by:

```text
python scripts/certify_cstar_fullsplit_stabilized_gate.py \
  --source-json scripts/results/cstar_annulus_fullsplit_cap1000_polish_20260606.json \
  --previous-json scripts/results/cstar_annulus_boundclean_normalized_qp_probe_direct_eval_20260605.json \
  --hessian-json scripts/results/cstar_annulus_direct_hessian_fullsplit_cap1000_free_20260606.json \
  --target-value 0.31226427 \
  --output-json scripts/results/cstar_annulus_fullsplit_theorem_gate_verification_cap1000_20260606.json \
  --output-md references/CSTAR_ANNULUS_FULLSPLIT_THEOREM_GATE_VERIFICATION_CAP1000_20260606.md
```

The released-face Hessian was regenerated by:

```text
python scripts/analyze_cstar_annulus_block_schur_gate.py \
  --backend torch-cuda --torch-precision float64 --fft-workers 24 \
  --fft-grid 160 --scale-max 1000 \
  --direct-hessian-json scripts/results/cstar_annulus_fullsplit_cap1000_polish_20260606.json \
  --direct-hessian-free-only --direct-hessian-step 0.01 \
  --direct-hessian-output-json scripts/results/cstar_annulus_direct_hessian_fullsplit_cap1000_free_20260606.json \
  --direct-hessian-md references/CSTAR_ANNULUS_DIRECT_HESSIAN_FULLSPLIT_CAP1000_FREE_20260606.md
```

The TeX insert compiles with:

```text
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=references references/CSTAR_ANNULUS_CAP1000_MANUSCRIPT_INSERT_20260606.tex
```

### Optional Full-Regeneration Tier

The minimal bundle above verifies the finite theorem from the archived
certificate matrices.  A larger full-regeneration archive may additionally
include the direction caches and coefficient files needed to rebuild the
candidate and Hessian from scratch.  That larger bundle is useful for long-term
reproducibility, but it is not necessary for the paper's proof text.

## What Changed From The Previous Packet

The previous best tracked value was

```text
0.31226424973923933.
```

A curvature-normalized QP probe improved the actual direct CUDA FFT ratio to

```text
0.31226426322102052.
```

That point had `121` coordinates at the artificial `scale_max=200` cap, so it
was not proof-ready.  Releasing the cap to `1000` and polishing gave

```text
0.31226426322102058,
```

an additional gain of only `5.551115123125783e-17`.  The maximum scale is
`200.00000000868158`, far below the cap `1000`, so the active cap obstruction
is removed.

## Gate Summary

| gate | status | evidence |
|---|---|---|
| cap release | closed | `cap_hit_count = 0` at `scale_max=1000` |
| finite replay value | closed with headroom | target margin `6.778979433352816e-09` |
| lower-bound KKT signs | closed | `154` zero-bound coordinates, positive zero gradients `0` |
| upper-bound KKT signs | vacuous | `0` upper-bound coordinates |
| free stationarity | closed to proof tolerance | free gradient max abs `3.963733163849133e-12` |
| active curvature | closed on released face | `647` Hessian columns recomputed at cap-1000 source |
| structural zero rows | closed | `6` zero-Hessian rows, `0` one-high triads |
| rounding ledger | closed by gamma budget | display target allows roughly `1.935e8` numerator operations before active residual |
| far-tail interface | optional stronger theorem | finite table already reaches `|k|^2=4*565=2260`; beyond this is multi-high/full-field tail work |

## Candidate Data

| quantity | value |
|---|---:|
| finite shell-split groups | `801` |
| free coordinates at cap 1000 | `647` |
| lower-bound coordinates | `154` |
| upper-bound coordinates | `0` |
| maximum scale | `200.00000000868158` |
| minimum positive scale | `0.019012003095923116` |
| final gradient norm | `2.5274837866370627e-10` |
| free gradient max abs | `3.963733163849133e-12` |
| zero-bound gradient max | `-3.5345695459004496e-13` |

## Hessian Certificate

The released-face Hessian was recomputed directly from the cap-1000 source:

```text
python scripts/analyze_cstar_annulus_block_schur_gate.py \
  --backend torch-cuda --torch-precision float64 --fft-workers 24 \
  --fft-grid 160 --scale-max 1000 \
  --direct-hessian-json scripts/results/cstar_annulus_fullsplit_cap1000_polish_20260606.json \
  --direct-hessian-free-only --direct-hessian-step 0.01 \
  --direct-hessian-output-json scripts/results/cstar_annulus_direct_hessian_fullsplit_cap1000_free_20260606.json \
  --direct-hessian-md references/CSTAR_ANNULUS_DIRECT_HESSIAN_FULLSPLIT_CAP1000_FREE_20260606.md
```

It computed `647` free columns in `276.522` seconds on CUDA.  The raw selected
maximum eigenvalue is numerically zero because the six final rows have exactly
zero one-high interactions.  After removing those structural zero rows, the
nonzero active block is strictly negative:

| threshold | kept | dropped | max eigenvalue |
|---:|---:|---:|---:|
| `1e-30` | `641` | `6` | `-3.0235227279516019e-23` |
| `1e-18` | `616` | `31` | `-1.3636744180048693e-18` |
| `1e-14` | `478` | `169` | `-1.045767926727662e-14` |

The curvature-normalized negative Hessian has a robust row-sum margin:

| threshold | kept | normalized min eig | Gershgorin margin | entry-radius target |
|---:|---:|---:|---:|---:|
| `1e-30` | `641` | `0.69033821039725485` | `0.50538223782086966` | `0.00039482987329755441` |
| `1e-18` | `616` | `0.69034517372077708` | `0.50609049428470598` | `0.00041145568641033007` |
| `1e-14` | `478` | `0.69052990004257697` | `0.51777064985618759` | `0.00054273653024757614` |

## Structural Zero Rows

The six zero-curvature rows are not a hidden flat direction.  The cutoff-zero
audit finds:

```text
cutoff rows                = 6
total one-high triads      = 0
total shared one-high triads = 0
total linear abs           = 5.5965044354606178e-49
total delta-X abs          = 8.1022798828489255e-50
total delta-D abs          = 2.8527750525380219e-48
structural_zero_pass       = True
```

These rows are handled by a structural no-triad lemma, not by a fragile small
eigenvalue estimate.

## Rounding And Residual Budget

The target `0.31226427` has enough room for both the residual Newton budget and
standard finite replay roundoff.

| quantity | value |
|---|---:|
| target minus candidate | `6.778979433352816e-09` |
| denominator gamma count | `4838` |
| numerator gamma count before residual | `195533441` |
| tight residual-plus-target row | threshold `1e-16`, kept/dropped `577/70` |
| remaining slack after tight residual row | `5.9791946083210946e-09` |
| numerator gamma count after tight residual row | `172463812` |

The finite replay proof should invoke the standard floating-point lemma

```text
gamma_m = m u / (1 - m u),   u = 2^-53,
```

or an outward-rounded interval replay.  The existing ledger shows that the
displayed target survives `gamma_1e7` with slack `6.4321286995963108e-09`.
For manuscript use, cite the theorem-gate JSON as the finite ledger and state
the arithmetic model explicitly.

## Manuscript Placement

Recommended placement in Paper 2:

1. Put this after the existing `k=2` and `k=3` finite-block certificates, in the
   section that discusses `C^*` and annulus/global-field structure.
2. Present it as the large finite annulus certificate, not as another scan.
3. Reference it from Paper 1 only as the current numerical value of the
   cancellation constant if the Paper 1 constants section needs the polish.

Recommended claim wording:

```text
The finite one-high annulus certificate gives
C^*_{ann,finite} <= 0.31226427, with certified candidate value
0.31226426322102058.
```

Use a stronger tail-stability claim only if the optional far-tail proposition
is included:

```text
Together with the far-tail exclusion beyond |k|^2=2260, the cap-1000
candidate is stable against the specified multi-high/full-field tail class at
the same displayed value.
```

## Do Not Cite

Do not cite the prefix `700`, `900`, or cap-200 exploratory scans as support for
the final value.  They were useful discovery tools, but the paper-facing
certificate is the cap-1000 full finite shell-split packet above.
