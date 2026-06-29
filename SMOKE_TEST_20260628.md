# Reproducibility Smoke Test

Date: 2026-06-28

This note records a short local smoke test of the public certificate archive.
It is not a replacement for the paper proofs; it verifies that the documented
public-facing reproducibility commands and integrity checks run on a clean local
checkout.

## Environment

```text
Python 3.12.4
```

Required imports for the documented Paper 2 certificate check succeeded:

```text
numpy, scipy, mpmath, sympy
```

Git LFS object integrity check:

```text
git lfs fsck
Git LFS fsck OK
```

## Paper 1

Paper 1 is artifact-only in the public archive.  The rendered C-star figure PDFs
and PNGs are included under `paper1/scripts/paper_figs/`.

## Paper 2

Command:

```bash
cd paper2
python scripts/certify_cstar_fullsplit_stabilized_gate.py \
  --source-json scripts/results/cstar_annulus_fullsplit_cap1000_polish_20260606.json \
  --previous-json scripts/results/cstar_annulus_boundclean_normalized_qp_probe_direct_eval_20260605.json \
  --hessian-json scripts/results/cstar_annulus_direct_hessian_fullsplit_cap1000_free_20260606.json \
  --target-value 0.31226427 \
  --output-json ../tmp_cstar_gate_verify_smoke.json \
  --output-md ../tmp_cstar_gate_verify_smoke.md
```

Result:

```text
ratio=0.31226426322102058
target_minus_ratio=6.779e-09
zero_positive_count=0
active_gradient_max_abs=3.964e-12
```

Temporary smoke-test output files were removed after the run.

## Paper 3

Command:

```powershell
Get-Content paper3\MANIFEST.sha256 | ForEach-Object {
  # verify each listed SHA-256 against the current file
}
```

Result:

```text
MANIFEST OK
```

## Python Bytecode Check

Command:

```bash
python -m compileall -q paper2/scripts
```

Result: completed successfully.
