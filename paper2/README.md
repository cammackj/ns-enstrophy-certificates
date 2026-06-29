# Paper 2 Certificates

This folder contains the public support package for Paper 2,
`Fourier Cancellation, Shell Taxonomy, and Sharp Constants for the
Navier--Stokes Enstrophy Coupling`.

Paths listed in the Paper 2 manuscript are relative to this folder.  For
example, the manuscript path
`scripts/results/cstar_annulus_fullsplit_cap1000_polish_20260606.json`
corresponds to:

```text
paper2/scripts/results/cstar_annulus_fullsplit_cap1000_polish_20260606.json
```

## Cap-1000 Annulus Certificate

The load-bearing cap-released finite one-high annulus artifacts are:

```text
scripts/results/cstar_annulus_fullsplit_cap1000_polish_20260606.json
scripts/results/cstar_annulus_direct_hessian_fullsplit_cap1000_free_20260606.json
scripts/results/cstar_annulus_fullsplit_theorem_gate_verification_cap1000_20260606.json
```

The theorem-gate verifier is:

```text
scripts/certify_cstar_fullsplit_stabilized_gate.py
```

Rerunning the theorem-gate verifier also uses the six cutoff-zero direction
caches under:

```text
scripts/results/w10_split_rank901_1413/
```

The direct FFT/Hessian support script is:

```text
scripts/analyze_cstar_annulus_block_schur_gate.py
```

Readable proof packets and manuscript insert drafts are under `references/`.

## K3 Algebraic And KKT Certificate

The `k=3` block certificate is summarized by:

```text
results/k3_w_primitive_polynomial.txt
results/k3_global_kkt_exclusion_ledger_20260531.json
references/K3_CERTIFICATE_PACKET.md
references/K3_CLOSED_FORM.md
references/K3_GLOBAL_KKT_EXCLUSION.md
```

The immediate source ledgers referenced by the global KKT exclusion ledger are
also archived under `results/`.

## Dyadic Shell Comparison Support

The finite low-block logs supporting the manuscript table are under `logs/`.
Use the later April 23 k=5 certificate, not the older partial April 11 k=5
provenance log:

```text
logs/certify_block_maximum_gpu_k4_full_234550.txt
logs/certify_block_maximum_gpu_k5_full_164635.txt
logs/certify_block_maximum_gpu_k6_full_121654.txt
logs/certify_block_maximum_gpu_k7_full_103450.txt
```

The `k=2` value used in Paper 2 is the exact six-mode closed form derived in
the manuscript:
`(3*sqrt(38)-5*sqrt(6))/(72*sqrt(7+sqrt(57))) = 0.022741865409341...`.

The finite and tail comparison proof for `k=8` onward is summarized by:

```text
references/PAPER_THEOREM_LEDGER.md
references/P11_CANCELLATION_BOUND.md
references/K11_SHELL_RHO_LEMMA.md
references/SPECTRAL_TAXONOMY_TAIL_REPAIR.md
```

The corresponding audit scripts are:

```text
scripts/gap3/route_b_centered_kernel_cert.py
scripts/gap3/centered_row_mp_verify.py
scripts/gap3/centered_row_coarse_cert.py
scripts/gap3/route_a_joint_kernel_cert.py
scripts/gap3/k11_row_mp_verify.py
scripts/gap3/k11_row_interval_screen.py
scripts/gap3/k11_row_coarse_cert.py
scripts/gap3/tail_cert.py
```

The Case-A `rho(13)=1` arithmetic-decoupling certificate ledger is:

```text
references/CASE_A_RHO13_DECOUPLING.md
scripts/gap3/gap3_arithmetic_decoupling.py
```

## Dependencies

Install dependencies from this directory with:

```bash
pip install -r requirements.txt
```
