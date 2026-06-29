# K45 Transition Certificate Package

This package is generated from existing finite ledgers; it is not a new broad scan.

## Headline

- Certificate-form status: `ready_for_manuscript_review`
- Branch target: `rho45=5e-05`, `p45=0.016`
- Displayed branch bound: `0.015375505962416292`
- Displayed branch margin: `0.0006244940375837085`
- All-15 branch target violations: `0`

## Local Handoff Split

- Total shell-local handoffs: `1718`
- True two-mode charts: `15`
- One-mode-only split charts: `1672`
- Zero-mode split charts: `31`

The analytic product penalties are:

```text
true two-mode local:       0.7071067811865475
one-mode-only split:      0.5
zero-mode split:          0.35355339059327373
```

## Branch Lemma Ledger

```text
all-15 tube candidates:          92
all-15 target violations:        0
all-15 max signedCross:          0.014231631541764942
all-15 margin:                   0.0017683684582350586
focused max signedCross:         0.015375505962416292
focused margin:                  0.0006244940375837085
displayed bound source:          focused_worst_chart_replay
```

Worst focused row:

```text
chart: 17 38 61 -> 17 38 49
rank: ascent:max_danger_branch_psi_penalty_5000
signed_cross: 0.015375505962416292
abs_Psi45: 2.4537745664824848e-05
R: 0.00017819606915008872
full_DPsi45_F: 0.1802337876131613
```

## Nonlocal Split

- Nonlocal handoff pairs: `144482`
- Shared-zero pairs: `114595`
- Shared-one pairs: `29887`
- Max side split product penalty: `0.7071067811865476`
- Passes `1/sqrt(2)` split bound: `True`

## Proof Gates

| gate | status | evidence |
| :--- | :--- | :--- |
| support-atlas separation | closed_certificate_form | Finite W45 catalogue separates endpoint, k4-only, k5-only, overlap, and remainder roles; support role counts={'bridge_k4_selector_side_only': 172, 'bridge_k5_core_side_only': 850, 'bridge_overlap_family_only': 195, 'bridge_remainder_no_transfer_family': 428, 'endpoint_chart': 822}. |
| local zero-mode split charts | closed_analytic_product_loss | 31 local shell handoffs have no shared concrete mode; product penalty <= 0.35355339059327373. |
| local one-mode-only split charts | closed_analytic_product_loss | 1672 local shell handoffs have one-mode but no two-mode witness; product penalty <= 0.5. |
| local true two-mode branch charts | closed_certificate_form | 15 true two-mode charts; branch target rho=5e-05, p=0.016; displayed bound=0.015375505962416292 from focused_worst_chart_replay; margin=0.0006244940375837085; all-15 violations=0. |
| nonlocal mass-splitting charts | closed_certificate_form | 144482 nonlocal handoff pairs split by shared-shell count; max side split product penalty=0.7071067811865476. |
| overlap and remainder charts | closed_by_transfer_coordinate | 623 bridge supports are overlap-only or outside-transfer-family remainder; overlap families cancel from Psi45 and remainders carry no endpoint transfer family. |

## Theorem-Ready Statement

On the finite K45 transition atlas, the transfer coordinate
`Psi45=S_{Sigma5_core}-S_{Sigma4_sel}` has no stationary,
endpoint-oriented, enstrophy-producing two-mode local bridge above
`p45=0.016` inside the displayed `rho45=5e-5` crossing tube at
certificate-form level.  The remaining local and nonlocal bridge
families are covered by product-loss or transfer-coordinate
cancellation alternatives.

For manuscript use, this is ready as a finite certificate theorem if
the paper cites the generated ledgers.  An artifact-free presentation
would still require expanding the finite ledger checks into interval
or rational arithmetic in the text.
