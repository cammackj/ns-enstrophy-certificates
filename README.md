# ns-enstrophy-certificates

Public reproducibility package for the companion Navier--Stokes enstrophy
papers by J. D. Cammack.

This repository is a curated certificate archive, not the full private research
workspace.  It contains the scripts, ledgers, and human-readable proof packets
needed to inspect or reproduce the paper-facing computer-assisted claims.  Large
exploratory scans, superseded trial runs, caches, and debugging artifacts are
intentionally omitted.

## Frozen Submission Archive

For the June 2026 manuscript package, use the annotated Git tag
`paper3-certificates-v3.0` as the frozen certificate archive.  This is the
history-cleaned, smoke-tested certificate snapshot cited by the current
submission package.

That tag fixes the certificate/result files, logs, and manifests used by the
submitted manuscripts.  The repository may continue to evolve after that tag:
scripts can be improved, documentation and provenance notes can be clarified,
and new checks can be added on later commits.  Any later certificate values or
load-bearing provenance changes should be cited through a new tag rather than by
moving the submitted archive.

In short: referees should audit the tagged archive; ongoing development should
continue on later commits or later tags.  Earlier public tags are superseded by
this curated archive snapshot.

## Layout

```text
paper1/
  scripts/paper_figs/    rendered figure assets used by Paper 1

paper2/
  scripts/               certificate and verification scripts
  results/               load-bearing certificate ledgers and value files
  references/            readable proof packets and manuscript insert drafts
  logs/                  finite low-block certificate logs supporting Paper 2

paper3/
  results/               load-bearing dynamic alignment certificate ledgers
  README.md              Paper 3 certificate index and artifact-free status
  MANIFEST.sha256        SHA-256 hashes for mirrored Paper 3 artifacts

MANIFEST.sha256          SHA-256 hashes for the curated archive snapshot
```

When Paper 2 lists artifact paths such as
`scripts/results/cstar_annulus_fullsplit_cap1000_polish_20260606.json`, read
those paths relative to `paper2/` in this repository.

## Dependencies

Install the combined Python dependencies from the repository root:

```bash
pip install -r requirements.txt
```

For paper-specific environments, use `paper2/requirements.txt`.

Some Paper 2 scripts can use CUDA through PyTorch, but the archived certificate
ledgers can be inspected without rerunning the long GPU/CPU searches.

The largest Paper 3 ledgers are stored with Git LFS.  Use a normal Git LFS
clone or run `git lfs pull` after cloning before inspecting those files.

## Paper 1 Support

Paper 1 public support files live under `paper1/`.  This public archive keeps
only the rendered C-star figure assets used by the manuscript.  Private Paper 1
development code, tests, exploratory logs, and figure-generation work products
are intentionally not part of this public certificate archive.

The rendered C-star trajectory figures used by the manuscript are archived in
`paper1/scripts/paper_figs/`.

## Paper 2 Certificates

Paper 2 support files live under `paper2/`.

The cap-released finite one-high annulus certificate is carried by:

```text
paper2/scripts/results/cstar_annulus_fullsplit_cap1000_polish_20260606.json
paper2/scripts/results/cstar_annulus_direct_hessian_fullsplit_cap1000_free_20260606.json
paper2/scripts/results/cstar_annulus_fullsplit_theorem_gate_verification_cap1000_20260606.json
```

The theorem-gate verifier can be rerun from `paper2/`:

```bash
cd paper2
python scripts/certify_cstar_fullsplit_stabilized_gate.py \
  --source-json scripts/results/cstar_annulus_fullsplit_cap1000_polish_20260606.json \
  --previous-json scripts/results/cstar_annulus_boundclean_normalized_qp_probe_direct_eval_20260605.json \
  --hessian-json scripts/results/cstar_annulus_direct_hessian_fullsplit_cap1000_free_20260606.json \
  --target-value 0.31226427 \
  --output-json scripts/results/cstar_annulus_fullsplit_theorem_gate_verification_cap1000_20260606.json \
  --output-md references/CSTAR_ANNULUS_FULLSPLIT_THEOREM_GATE_VERIFICATION_CAP1000_20260606.md
```

The exact/algebraic and finite KKT support package for the `k=3` block is
archived in:

```text
paper2/results/k3_w_primitive_polynomial.txt
paper2/results/k3_global_kkt_exclusion_ledger_20260531.json
paper2/references/K3_CERTIFICATE_PACKET.md
paper2/references/K3_CLOSED_FORM.md
paper2/references/K3_GLOBAL_KKT_EXCLUSION.md
```

The archive intentionally omits superseded exploratory scans, private working
notes, and duplicate legacy logs that are not needed for the paper-facing audit.

## Paper 3 Certificates

Paper 3 support files live under `paper3/`.

The mirrored Paper 3 archive contains only the load-bearing certificate-form
ledgers cited by the manuscript: the `K34`, `K45`, `K56`, `K67`, and all-later
atlas ledgers.  Exploratory scans and private working notes are intentionally
omitted.

See:

```text
paper3/README.md
paper3/MANIFEST.sha256
```

## Provenance

These files were extracted from a larger private research repository to provide
a stable, citable public record for the verification results referenced in the
papers.  The private repository contains additional exploratory scripts,
intermediate results, and work in progress that are not part of the
paper-facing certificate archive.

## Questions and Issues

Please open issues at:

https://github.com/cammackj/ns-enstrophy-certificates/issues
