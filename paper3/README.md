# Paper 3 Certificates

This folder mirrors the load-bearing certificate-form ledgers cited by Paper 3,
`Dynamic Spectral Alignment`.

The archive is deliberately narrow.  It contains the finite ledgers used by the
manuscript route and omits exploratory scans, superseded runs, caches, and
private working notes.

## Load-Bearing Ledgers

The Paper 3 manuscript cites these directories in the certificate-form
dependency ledger:

```text
results/20260613_130427_audit_k34_transition_cover_k34_transition_cover_audit_v28_endpoint_packaged_paper_ready/
results/20260614_000621_package_k45_transition_certificate_k45_transition_certificate_package_v1/
results/20260614_094249_package_k45_transition_certificate_k56_transition_certificate_package_v1/
results/20260614_132544_calibrate_type_signature_envelopes_k67_shell_role_prefilter_v1/
results/20260614_135301_calibrate_type_signature_envelopes_k67_handoff_reduction_v1/
results/20260614_141011_autodiff_k4_selector_derivatives_k67_tangent_k6_s6core_v1/
results/20260614_141026_autodiff_k4_selector_derivatives_k67_tangent_k6_s7core_v1/
results/20260614_141041_autodiff_k4_selector_derivatives_k67_tangent_k7_s6core_v1/
results/20260614_141149_autodiff_k4_selector_derivatives_k67_tangent_k7_s7core_v1/
results/20260614_153838_calibrate_type_signature_envelopes_k67_mode_refinement_overlap_audit_v2_full_pairs/
results/20260616_033333_audit_all_later_atlas_all_later_atlas_audit_certificate_form_route_closed_final2/
```

The first three directories package the finite transition certificates for
`K34`, `K45`, and `K56`.  The next seven package the `K67` shell-role, handoff,
tangent-replay, and concrete-refinement ledgers.  The final directory packages
the all-later atlas audit used by the certificate-form route.

## Manifest

`MANIFEST.sha256` records a SHA-256 hash for every mirrored Paper 3 artifact,
using paths relative to this repository root.

The largest K67 ledgers are stored with Git LFS.  If a clone contains pointer
files instead of the full artifacts, run `git lfs pull` from the repository
root.

## Artifact-Free Status

These files support the certificate-form version of the Paper 3 route.  A fully
artifact-free presentation would replace the finite generated/rounded ledgers by
outward-rounded interval or rational enclosures inside the manuscript.
