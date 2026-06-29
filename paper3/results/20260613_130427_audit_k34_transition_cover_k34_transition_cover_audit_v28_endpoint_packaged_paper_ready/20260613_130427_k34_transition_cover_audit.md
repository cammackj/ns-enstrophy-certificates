# K34 Transition Cover Audit

This ledger is generated from existing Paper 3 artifacts.  It is not a new scan.

## Headline

- V13 corrected endpoint gap: `0.010155739007683441`
- V13 quotient gate passes: `True`
- Face-ledger rows: `20`

Caveat: `K34 signature` in the face-class ledger is a triad-row flag
for the broader K34 signature ledger, not just the three selected-family
signatures in `S4`.  The no-selector exclusion needed for the theorem is
support-level and must not be inferred from a single outside-ledger row in
a support that also has selector-bearing rows.

## Proof Obligations

| item | status | next action / evidence |
| :--- | :--- | :--- |
| Define K34 finite tube | defined_in_k34_finite_atlas_package | The finite window W34={8,...,31}, normalized strip -b4<=G4<=a3, residual tube delta_res<=0.03, bridge boundary charts, and residual-boundary charts are stated in the K34 finite-atlas package. |
| Define endpoint chart interiors U3 and U4 | endpoint_charts_certificate_form_packaged | U3 is imported from the Paper 2 exact k=3 Krawczyk/KKT chart; U4 is the Paper 3 radius-0.30 full finite k4 chart with G4<=-2.57935480298416e-05 and DG4[F]<=-0.31253357722405578. |
| V13 selector-bearing residual face | closed | corrected endpoint gap 1.015573900768344112e-02; quotient gate passes=True |
| Five raw-below selector-bearing supports | raw_below_tube_exterior_rounded_certificate_packaged | 5 logs parsed; all below endpoint=True; all global replay pass=True; minimum certified-upper gap=0.000437882905546775 on support 10,13,19. Boundary ledger: 10/10 boundary faces parsed; complete=True; all below endpoint=True; minimum boundary gap=0.0029449843956699945 on 13 22 29 upper. Interior ledger: Interior diagnostic paper3\results\20260613_021357_certify_k34_raw_below_interior_quotient_k34_raw_below_interior_quotient_v8_abs_satellite_tube_cuda\20260613_021357_raw_below_interior_quotient_summary.json; inactive loga signs nonpositive=True; active quotient strictly negative=False; max quotient Hessian eig=1.2166449216153287e-07; max satellite X fraction=0.0001319602898723904; max positive satellite correction=0.0; one-satellite channels vanish=True; satellite release Hessians positive=True; min cubic half-gap tube delta=0.000398305417031442; min absolute two-plus half-gap tube delta=5.0595518674385376e-05; max observed fraction of absolute two-plus tube=0.6190826003129184; min endpoint gap=0.00043788290554677353. Exterior ledger: Exterior screen paper3\results\20260613_021837_screen_k34_selector_strip_k34_selector_strip_raw_below_exterior_tube_v1\20260613_021837_selector_strip_summary.json; all excluded numerically=True; minimum exterior gap=0.0029441406987582817; satellite exterior only=True. Next: State the raw-below lemma as a rounded certificate-form tube/exterior package.  For final theorem-grade form, intervalize the satellite-tube coefficient ledger and replace the exterior screen by an outward interval certificate. |
| No-selector support exclusion | closed_by_residual_support_ledger | The full residual support ledger has 278 supports: 219 have no selected-family signature, so S4=0 and strip membership gives R=G4<=a3. The analytic endpoint gap is 0.009637796977544695. |
| Skeleton bridge cover | bridge_charts_rounded_certificate_packaged | The skeleton bridge ledger has 57 supports: 47 are closed by S4=0 and strip membership, leaving 10 selector-bearing skeleton supports. Skeleton survivor screen paper3\results\20260613_030852_screen_k34_selector_strip_k34_skeleton_selector_survivors_float64_v1\20260613_030852_selector_strip_summary.json; above-endpoint screen survivors=4. Rounded bridge certificate paper3\results\20260613_074717_audit_k34_bridge_tangent_k34_bridge_boundary_tangent_rounded_certificate_v1\20260613_074717_k34_bridge_tangent_audit.json; rows=6; all escape signs certified=True; minimum certified radius=0.01136467284454005 on 10 21 29 upper. Next: For final theorem-grade form, replace the rounded float derivative budgets by interval/rational outward enclosures; no new bridge search is indicated. |
| Residual leakage/involution classification | residual_boundary_escape_charts_packaged | Classify the 59 selector-bearing residual supports by selected signature (C->C->B, A->C->C, A->D->B); the other residual supports are closed by S4=0. Single-band screen paper3\results\20260613_024533_screen_k34_selector_strip_k34_single_band_selector_strip_screen_v1\20260613_024533_selector_strip_summary.json; supports screened=53; above endpoint=17; within 1e-3 below=3. Minimum residual X-fraction among above-endpoint candidates=0.09440071723600187 on 10 20 22. This shows the scalar G4 strip alone is insufficient; the finite tube must include a residual-mass/leakage coordinate. Targeted residual tube screen paper3\results\20260613_080720_screen_k34_selector_strip_k34_residual_tube_above_endpoint_delta003_v1\20260613_080720_selector_strip_summary.json; delta_res<=0.03; supports=17; all excluded=True; minimum gap=0.0005542818761423256 on 8 16 24. Constrained residual-tube replay paper3\results\20260613_124730_certify_k34_residual_tube_k34_residual_tube_tight_8_16_24_certificate_v1\20260613_124730_residual_tube_certificate_summary.json; paper3\results\20260613_124901_certify_k34_residual_tube_k34_residual_tube_17support_certificate_v1\20260613_124901_residual_tube_certificate_summary.json; paper3\results\20260613_125000_certify_k34_residual_tube_k34_residual_tube_tight_9_10_25_10_20_22_certificate_v1\20260613_125000_residual_tube_certificate_summary.json; delta_res<=0.03; supports=17; above endpoint=3 (10 20 22, 8 16 24, 9 10 25); minimum positive gap among non-escape residual supports=0.0021348985824343088 on 8 22 26; max constraint violation=1.7140681651817147e-09. Residual-boundary tangent certificates parsed=3; all escape signs certified=True; minimum certified radius=0.036407184119710716 on 9 10 25. Next: State the residual-mass lemma as: below-endpoint residual supports are excluded by constrained replay, and the above-endpoint residual-boundary atoms are finite one-sided escape charts.  Final theorem-grade form should intervalize the rounded tangent budgets. |
| Positive transition-cost theorem | finite_atlas_certificate_form_ready_interval_polish_pending | Integrate the theorem insert and decide the presentation standard: certificate-form with repository ledgers, or fully outward-rounded interval enclosures for the rounded bridge, residual, raw-below, and dense k4 tensor inputs. |

## Skeleton Bridge Support Split

- Skeleton bridge supports: `57`
- Selector-bearing skeleton supports: `10`
- No-selector skeleton supports: `47`
- Analytic no-selector endpoint gap in strip: `0.009637796977544695`

Selector-bearing skeleton supports:

| support | selected-family triads | selected-family signatures | status |
| :--- | ---: | :--- | :--- |
| 14 25 29 | 288 | A->C->C:288 | selector_bearing_skeleton_support_requires_endpoint_chart_or_certificate |
| 10 17 21 | 192 | C->C->B:192 | selector_bearing_skeleton_support_requires_endpoint_chart_or_certificate |
| 10 21 29 | 192 | C->C->B:192 | selector_bearing_skeleton_support_requires_endpoint_chart_or_certificate |
| 14 17 17 | 192 | A->C->C:192 | selector_bearing_skeleton_support_requires_endpoint_chart_or_certificate |
| 14 17 29 | 192 | A->C->C:192 | selector_bearing_skeleton_support_requires_endpoint_chart_or_certificate |
| 10 20 26 | 144 | A->C->C:144 | selector_bearing_skeleton_support_requires_endpoint_chart_or_certificate |
| 10 10 26 | 96 | A->C->C:96 | selector_bearing_skeleton_support_requires_endpoint_chart_or_certificate |
| 10 25 27 | 96 | C->C->B:96 | selector_bearing_skeleton_support_requires_endpoint_chart_or_certificate |
| 10 27 29 | 96 | C->C->B:96 | selector_bearing_skeleton_support_requires_endpoint_chart_or_certificate |
| 14 17 25 | 96 | A->C->C:96 | selector_bearing_skeleton_support_requires_endpoint_chart_or_certificate |

Skeleton survivor screen:

- Summary JSON: `paper3\results\20260613_030852_screen_k34_selector_strip_k34_skeleton_selector_survivors_float64_v1\20260613_030852_selector_strip_summary.json`

| support | best R | G4 | endpoint gap |
| :--- | ---: | ---: | ---: |
| 10 21 29 | 2.19906636372781822e-02 | 5.94311235566874485e-03 | -9.26267031731407148e-04 |
| 10 20 26 | 3.73698599420302716e-02 | -4.12002895878329362e-03 | -1.63054633364834965e-02 |
| 10 26 | 3.62579084115207256e-02 | -4.98603288523570554e-03 | -1.51935118059739506e-02 |
| 10 25 27 | 2.21960377219897406e-02 | 1.14200576817892752e-02 | -1.13164111644296558e-03 |

Skeleton bridge rounded certificate:

- Summary JSON: `paper3\results\20260613_074717_audit_k34_bridge_tangent_k34_bridge_boundary_tangent_rounded_certificate_v1\20260613_074717_k34_bridge_tangent_audit.json`
- Certified rows: `6`
- All escape signs certified: `True`
- Minimum certified radius: `0.01136467284454005` on `10 21 29 upper`
- Slack/padding: value `0.02`, grad `0.02`, Hessian `0.05`, radius factor `0.9`

| chart | G4 | DG4[F] | certified radius | interpretation |
| :--- | ---: | ---: | ---: | :--- |
| 10 20 26 lower | -5.02801345876217193e-03 | 6.47751719606352916e-01 | 6.85974840517601520e-02 | escapes toward k3-side positive G4 |
| 10 20 26 upper | 1.14265996565059806e-02 | -1.94433188576930593e+00 | 9.40185964614199260e-02 | escapes toward k4-side negative G4 |
| 10 21 29 upper | 1.14265996279847206e-02 | -1.08487234422264628e-01 | 1.13646728445400500e-02 | escapes toward k4-side negative G4 |
| 10 25 27 upper | 1.14265996175230526e-02 | 7.09392422498092645e-01 | 4.58154656115119380e-02 | escapes toward k3-side positive G4 |
| 10 26 lower | -5.02801771165826278e-03 | -2.95453247684107356e-01 | 5.05170523099966878e-02 | escapes toward k4-side negative G4 |
| 10 26 upper | 1.14265996285331881e-02 | -1.80217960480247491e-01 | 2.71490411850437603e-02 | escapes toward k4-side negative G4 |

## Mixed Support Split

- Mixed residual shell supports: `58`
- Selector-bearing supports: `6`
- No-selector supports: `52`
- Analytic no-selector endpoint gap in strip: `0.009637796977544695`

Selector-bearing supports:

| support | selected-family triads | selected-family signatures | status |
| :--- | ---: | :--- | :--- |
| 13 17 30 | 192 | C->C->B:192 | selector_bearing_support_requires_certificate_or_existing_closure |
| 13 29 30 | 192 | C->C->B:192 | selector_bearing_support_requires_certificate_or_existing_closure |
| 10 13 19 | 96 | C->C->B:96 | selector_bearing_support_requires_certificate_or_existing_closure |
| 13 17 24 | 96 | A->C->C:96 | selector_bearing_support_requires_certificate_or_existing_closure |
| 13 22 25 | 96 | C->C->B:96 | selector_bearing_support_requires_certificate_or_existing_closure |
| 13 22 29 | 96 | C->C->B:96 | selector_bearing_support_requires_certificate_or_existing_closure |

## All Residual Support Split

- Residual shell supports: `278`
- Selector-bearing residual supports: `59`
- No-selector residual supports: `219`
- Analytic no-selector endpoint gap in strip: `0.009637796977544695`

Top selector-bearing residual supports:

| support | selected-family triads | residual bands | selected-family signatures | status |
| :--- | ---: | :--- | :--- | :--- |
| 17 29 30 | 384 | i4_residual_only:1152 | C->C->B:384 | selector_bearing_residual_support_requires_certificate_or_taxonomy |
| 9 20 29 | 288 | i3_residual_only:864 | C->C->B:288 | selector_bearing_residual_support_requires_certificate_or_taxonomy |
| 10 17 19 | 192 | i4_residual_only:576 | C->C->B:192 | selector_bearing_residual_support_requires_certificate_or_taxonomy |
| 10 19 29 | 192 | i4_residual_only:576 | C->C->B:192 | selector_bearing_residual_support_requires_certificate_or_taxonomy |
| 13 17 26 | 192 | i3_residual_only:576 | A->C->C:192 | selector_bearing_residual_support_requires_certificate_or_taxonomy |
| 13 17 30 | 192 | mixed_i3_i4_residual:576 | C->C->B:192 | selector_bearing_residual_support_requires_certificate_or_taxonomy |
| 13 29 30 | 192 | mixed_i3_i4_residual:576 | C->C->B:192 | selector_bearing_residual_support_requires_certificate_or_taxonomy |
| 17 24 29 | 192 | i4_residual_only:576 | A->C->C:192 | selector_bearing_residual_support_requires_certificate_or_taxonomy |
| 24 25 29 | 192 | i4_residual_only:576 | A->C->C:192 | selector_bearing_residual_support_requires_certificate_or_taxonomy |
| 29 29 30 | 192 | i4_residual_only:576 | C->C->B:192 | selector_bearing_residual_support_requires_certificate_or_taxonomy |
| 9 10 29 | 192 | i3_residual_only:576 | C->C->B:192 | selector_bearing_residual_support_requires_certificate_or_taxonomy |
| 13 25 26 | 144 | i3_residual_only:432 | A->C->C:144 | selector_bearing_residual_support_requires_certificate_or_taxonomy |
| 9 17 20 | 144 | i3_residual_only:432 | C->C->B:144 | selector_bearing_residual_support_requires_certificate_or_taxonomy |
| 10 11 13 | 96 | i3_residual_only:288 | C->C->B:96 | selector_bearing_residual_support_requires_certificate_or_taxonomy |
| 10 11 17 | 96 | i3_residual_only:288 | C->C->B:96 | selector_bearing_residual_support_requires_certificate_or_taxonomy |
| 10 11 25 | 96 | i3_residual_only:288 | C->C->B:96 | selector_bearing_residual_support_requires_certificate_or_taxonomy |
| 10 11 29 | 96 | i3_residual_only:288 | C->C->B:96 | selector_bearing_residual_support_requires_certificate_or_taxonomy |
| 10 13 19 | 96 | mixed_i3_i4_residual:288 | C->C->B:96 | selector_bearing_residual_support_requires_certificate_or_taxonomy |
| 10 13 27 | 96 | i3_residual_only:288 | C->C->B:96 | selector_bearing_residual_support_requires_certificate_or_taxonomy |
| 10 20 22 | 96 | i4_residual_only:288 | C->C->B:96 | selector_bearing_residual_support_requires_certificate_or_taxonomy |

## Residual Leakage Tube Budget

- Selector-bearing residual budgets: `59`
- Maximum one-residual-slot coefficient C1: `153.908762741836`
- Maximum two-residual-slot coefficient C2: `73.31495486294433`
- Maximum three-residual-slot coefficient C3: `0.0`
- Maximum coarse leakage at delta=0.01: `15.75149087186085`
- Maximum coarse leakage at delta=0.05: `36.218118570749475`
- Maximum coarse leakage at delta=0.10: `52.27637018903884`

| support | residual shells | C1 | C2 | C3 | leakage at 0.01 | leakage at 0.05 | leakage at 0.10 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 26 30 | 30 | 1.539088e+02 | 3.606146e+01 | 0.000000e+00 | 1.575149e+01 | 3.621812e+01 | 5.227637e+01 |
| 17 29 30 | 30 | 1.507021e+02 | 0.000000e+00 | 0.000000e+00 | 1.507021e+01 | 3.369801e+01 | 4.765618e+01 |
| 8 14 30 | 30 | 1.155996e+02 | 0.000000e+00 | 0.000000e+00 | 1.155996e+01 | 2.584885e+01 | 3.655579e+01 |
| 9 20 29 | 9 | 1.111281e+02 | 1.788433e+01 | 0.000000e+00 | 1.129166e+01 | 2.574322e+01 | 3.693023e+01 |
| 14 18 30 | 30 | 8.534273e+01 | 2.780843e+01 | 0.000000e+00 | 8.812357e+00 | 2.047364e+01 | 2.976858e+01 |
| 17 24 29 | 24 | 7.524233e+01 | 0.000000e+00 | 0.000000e+00 | 7.524233e+00 | 1.682470e+01 | 2.379371e+01 |
| 9 10 29 | 9 | 7.518045e+01 | 3.397253e+01 | 0.000000e+00 | 7.857770e+00 | 1.850949e+01 | 2.717140e+01 |
| 17 22 29 | 22 | 7.482410e+01 | 0.000000e+00 | 0.000000e+00 | 7.482410e+00 | 1.673118e+01 | 2.366146e+01 |
| 18 26 30 | 30 | 7.250040e+01 | 4.856570e+01 | 0.000000e+00 | 7.735697e+00 | 1.863987e+01 | 2.778321e+01 |
| 10 19 29 | 19 | 6.910962e+01 | 0.000000e+00 | 0.000000e+00 | 6.910962e+00 | 1.545338e+01 | 2.185438e+01 |
| 10 17 19 | 19 | 6.630689e+01 | 0.000000e+00 | 0.000000e+00 | 6.630689e+00 | 1.482667e+01 | 2.096808e+01 |
| 13 17 26 | 13 | 5.821957e+01 | 1.469139e+01 | 0.000000e+00 | 5.968871e+00 | 1.375286e+01 | 1.987978e+01 |
| 13 29 30 | 13 30 | 5.648686e+01 | 5.968947e+01 | 0.000000e+00 | 6.245580e+00 | 1.561532e+01 | 2.383166e+01 |
| 9 17 20 | 9 | 5.320369e+01 | 1.788433e+01 | 0.000000e+00 | 5.499212e+00 | 1.279092e+01 | 1.861292e+01 |
| 10 20 22 | 22 | 5.065215e+01 | 0.000000e+00 | 0.000000e+00 | 5.065215e+00 | 1.132617e+01 | 1.601762e+01 |
| 17 22 25 | 22 | 4.976743e+01 | 0.000000e+00 | 0.000000e+00 | 4.976743e+00 | 1.112834e+01 | 1.573784e+01 |
| 12 17 25 | 12 | 4.488819e+01 | 0.000000e+00 | 0.000000e+00 | 4.488819e+00 | 1.003730e+01 | 1.419489e+01 |
| 13 25 26 | 13 | 4.398186e+01 | 1.469139e+01 | 0.000000e+00 | 4.545100e+00 | 1.056921e+01 | 1.537742e+01 |
| 24 25 29 | 24 | 4.167235e+01 | 0.000000e+00 | 0.000000e+00 | 4.167235e+00 | 9.318221e+00 | 1.317796e+01 |
| 14 18 22 | 22 | 4.121590e+01 | 0.000000e+00 | 0.000000e+00 | 4.121590e+00 | 9.216156e+00 | 1.303361e+01 |

## Single-Band Selector Residual Screen

- Summary JSON: `paper3\results\20260613_024533_screen_k34_selector_strip_k34_single_band_selector_strip_screen_v1\20260613_024533_selector_strip_summary.json`
- Supports screened: `53`
- Above endpoint in scalar strip screen: `17`
- Within `1e-3` below endpoint: `3`
- Minimum residual `X` fraction among above-endpoint candidates: `0.09440071723600187` on `10 20 22`

Interpretation: the scalar `G4` strip alone is not the finite tube. The K34 tube needs a residual-mass/leakage coordinate.

| support | residual shells | residual X frac | best R | G4 | endpoint gap | mode |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| 8 16 24 | 24 | 1.62625617416142426e-01 | 4.23998832702636719e-02 | 1.14288423210382462e-02 | -2.13354866647168968e-02 | strip_lbfgs |
| 9 20 29 | 9 | 4.56888491546835140e-01 | 3.72623763978481293e-02 | -3.39780002832412720e-03 | -1.61979797923013542e-02 | strip_lbfgs |
| 9 10 25 | 9 | 3.35819077019026135e-01 | 3.68637852370738983e-02 | 1.14279240369796753e-02 | -1.57993886315271233e-02 | strip_lbfgs |
| 9 10 29 | 9 | 3.57739329081592672e-01 | 3.66613455116748810e-02 | 1.14277526736259460e-02 | -1.55969489061281059e-02 | strip_lbfgs |
| 10 11 29 | 11 | 3.67298115034353612e-01 | 3.34403701126575470e-02 | 1.14236380904912949e-02 | -1.23759735071107720e-02 | strip_lbfgs |
| 9 17 20 | 9 | 5.73499037634819686e-01 | 3.32286916673183441e-02 | 1.14310104399919510e-02 | -1.21642950617715691e-02 | strip_lbfgs |
| 10 19 29 | 19 | 3.28892282517735612e-01 | 3.31356599926948547e-02 | 1.14288982003927231e-02 | -1.20712633871480797e-02 | strip_lbfgs |
| 13 16 29 | 13 | 3.70411880910418656e-01 | 3.18594500422477722e-02 | 1.14276148378849030e-02 | -1.07950534367009972e-02 | strip_lbfgs |
| 10 11 25 | 11 | 3.61538738212260291e-01 | 3.11403106898069382e-02 | 1.14270281046628952e-02 | -1.00759140842601631e-02 | strip_lbfgs |
| 9 13 20 | 9 13 | 7.19800664547001201e-01 | 3.03942505270242691e-02 | 1.14291682839393616e-02 | -9.32985392147749407e-03 | strip_lbfgs |
| 12 13 29 | 12 13 | 6.89758272474305212e-01 | 2.97925453633069992e-02 | 7.07847438752651215e-03 | -8.72814875776022417e-03 | strip_lbfgs |
| 10 20 22 | 22 | 9.44007172360018659e-02 | 2.42208894342184067e-02 | 1.14366365596652031e-02 | -3.15649282867163164e-03 | strip_lbfgs |

## Residual Tube Targeted Screen

- Summary JSON: `paper3\results\20260613_080720_screen_k34_selector_strip_k34_residual_tube_above_endpoint_delta003_v1\20260613_080720_selector_strip_summary.json`
- Residual tube max delta: `0.03`
- Supports screened: `17`
- All excluded against endpoint: `True`
- Minimum endpoint gap: `0.0005542818761423256` on `8 16 24`

| support | residual shells | residual delta | best R | G4 | endpoint gap | mode |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| 8 16 24 | 24 | 2.98228897154331207e-02 | 2.05101147294044495e-02 | 8.66629462689161301e-03 | 5.54281876142325575e-04 | strip_lbfgs |
| 9 10 25 | 9 | 2.99999713897705078e-02 | 1.70172229409217834e-02 | 1.08611043542623520e-02 | 4.04717366462499159e-03 | strip_lbfgs |
| 8 22 26 | 22 | 2.99991648644208908e-02 | 1.69194228947162628e-02 | 1.12412003800272942e-02 | 4.14497371083051222e-03 | strip_lbfgs |
| 10 19 29 | 19 | 2.97828186303377151e-02 | 1.50041813030838966e-02 | 8.23536142706871033e-03 | 6.06021530246287840e-03 | strip_lbfgs |
| 10 20 22 | 22 | 2.61855553835630417e-02 | 1.32659124210476875e-02 | 1.14283561706542969e-02 | 7.79848418449908751e-03 | strip_lbfgs |
| 10 11 25 | 11 | 2.94631458818912506e-02 | 1.18323685601353645e-02 | 1.12252775579690933e-02 | 9.23202804541141051e-03 | strip_lbfgs |
| 9 10 29 | 9 | 1.60157885402441025e-02 | 1.14391557872295380e-02 | 1.14388940855860710e-02 | 9.62524081831723707e-03 | level_lbfgs |
| 10 11 29 | 11 | 2.47426051646471024e-03 | 1.14389294758439064e-02 | 1.14389657974243164e-02 | 9.62546712970286863e-03 | level_lbfgs |
| 9 20 29 | 9 | 2.99999937415122986e-02 | 9.32249613106250763e-03 | -3.83856054395437241e-03 | 1.17419004744842674e-02 | strip_lbfgs |
| 8 14 30 | 30 | 2.99999043345451355e-02 | 8.98055173456668854e-03 | 9.60456952452659607e-03 | 1.20838448709800865e-02 | strip_lbfgs |
| 13 16 29 | 13 | 2.99801770597696304e-02 | 7.88744539022445679e-03 | -4.99038025736808777e-03 | 1.31769512153223183e-02 | strip_lbfgs |
| 9 10 17 | 9 | 2.97571737319231033e-02 | 6.57153641805052757e-03 | -1.28414481878280640e-04 | 1.44928601874962475e-02 | strip_lbfgs |
| 9 17 20 | 9 | 2.99999881535768509e-02 | 4.80728782713413239e-03 | -3.83769534528255463e-03 | 1.62571087784126427e-02 | strip_lbfgs |
| 10 11 17 | 11 | 2.99999192357063293e-02 | 4.36592521145939827e-03 | 4.30140877142548561e-03 | 1.66984713940873768e-02 | strip_lbfgs |
| 13 26 29 | 13 | 2.99999956041574478e-02 | 3.81759088486433029e-03 | 1.14192543551325798e-02 | 1.72468057206824447e-02 | strip_lbfgs |
| 12 13 29 | 12 13 | 2.99999918788671494e-02 | 2.87244678474962711e-03 | -2.20757653005421162e-03 | 1.81919498207971479e-02 | level_lbfgs |
| 9 13 20 | 9 13 | 2.99999825656414032e-02 | 6.73751928843557835e-04 | 5.16491185408085585e-04 | 2.03906446767032172e-02 | strip_lbfgs |

## Constrained Residual Tube Certificate

- Summary JSON: `paper3\results\20260613_124730_certify_k34_residual_tube_k34_residual_tube_tight_8_16_24_certificate_v1\20260613_124730_residual_tube_certificate_summary.json; paper3\results\20260613_124901_certify_k34_residual_tube_k34_residual_tube_17support_certificate_v1\20260613_124901_residual_tube_certificate_summary.json; paper3\results\20260613_125000_certify_k34_residual_tube_k34_residual_tube_tight_9_10_25_10_20_22_certificate_v1\20260613_125000_residual_tube_certificate_summary.json`
- Residual tube max delta: `0.03`
- Supports replayed: `17`
- Above-endpoint residual-boundary atoms: `3`
- Above-endpoint supports: `10 20 22, 8 16 24, 9 10 25`
- Minimum positive endpoint gap outside escape atoms: `0.0021348985824343088` on `8 22 26`
- Maximum constraint violation: `1.7140681651817147e-09`

| support | residual shells | R | G4 | delta_res | endpoint gap | violation |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| 10 11 17 | 11 | 8.21237483506626238e-03 | 8.21087209966839260e-03 | 3.00000000008568586e-02 | 1.28520217704805127e-02 | 8.569e-13 |
| 10 11 25 | 11 | 1.49484588231880121e-02 | 1.14265996711056736e-02 | 2.99999997477304915e-02 | 6.11593778235876295e-03 | 4.310e-11 |
| 10 11 29 | 11 | 1.14265629353312893e-02 | 1.14265996280020782e-02 | 2.47350189143539893e-03 | 9.63783367021548570e-03 | 0.000e+00 |
| 10 19 29 | 19 | 1.83331461923274992e-02 | 1.14265994396366491e-02 | 3.00000003042156585e-02 | 2.73125041321927581e-03 | 3.042e-10 |
| 10 20 22 | 22 | 2.22166842589208088e-02 | 1.14265996280733719e-02 | 2.99999999999968070e-02 | -1.15228765337403380e-03 | 7.129e-14 |
| 12 13 29 | 12 13 | 5.97059420745879241e-03 | -4.21492171238967671e-03 | 2.99999492153439629e-02 | 1.50938023980879826e-02 | 0.000e+00 |
| 13 16 29 | 13 | 1.47335295697075958e-02 | -5.18837171278844697e-04 | 2.99999999812470414e-02 | 6.33086703583917929e-03 | 0.000e+00 |
| 13 26 29 | 13 | 6.40349722447940226e-03 | 6.40454258206742642e-03 | 2.99999999945798797e-02 | 1.46608993810673736e-02 | 0.000e+00 |
| 8 14 30 | 30 | 1.26519264659996945e-02 | 1.13884437744390041e-02 | 2.98587747598076762e-02 | 8.41247013954708051e-03 | 0.000e+00 |
| 8 16 24 | 24 | 3.52118820249046199e-02 | 1.14265996280406255e-02 | 2.99999999998385863e-02 | -1.41474854193578449e-02 | 3.855e-14 |
| 8 22 26 | 22 | 1.89294980231124663e-02 | 1.14266013420702452e-02 | 2.99999990755943531e-02 | 2.13489858243430877e-03 | 1.714e-09 |
| 9 10 17 | 9 | 9.76435944446123021e-03 | -5.02785340190977335e-03 | 2.99999998241027438e-02 | 1.13000371610855448e-02 | 0.000e+00 |
| 9 10 25 | 9 | 2.24437787342958092e-02 | 1.14265996279884607e-02 | 3.00000000000649296e-02 | -1.37938212874903418e-03 | 6.493e-14 |
| 9 10 29 | 9 | 1.14280123769276952e-02 | 1.14265748632908261e-02 | 1.85852706952406577e-02 | 9.63638422861907984e-03 | 0.000e+00 |
| 9 13 20 | 9 13 | 1.91191559984648144e-03 | 1.91174140480888021e-03 | 2.99999994857076055e-02 | 1.91524810057002934e-02 | 0.000e+00 |
| 9 17 20 | 9 | 7.50170190902312104e-03 | -5.02801784984557420e-03 | 3.00000000000056957e-02 | 1.35626946965236549e-02 | 5.697e-15 |
| 9 20 29 | 9 | 1.40843161500867620e-02 | -5.02801798763456362e-03 | 2.99999998010969209e-02 | 6.98008045546001307e-03 | 1.364e-10 |

## Residual-Boundary Escape Charts

- Certified supports: `10 20 22, 8 16 24, 9 10 25`
- All escape signs certified: `True`
- Minimum certified radius: `0.036407184119710716` on `9 10 25`

| support | R | G4 | DG4[F] | certified radius | interpretation |
| :--- | ---: | ---: | ---: | ---: | :--- |
| 10 20 22 | 2.22166842589208088e-02 | 1.14265996280733719e-02 | 9.67861987820524305e+00 | 2.57025527183587954e-01 | escapes toward k3-side positive G4 |
| 8 16 24 | 3.52118820249046199e-02 | 1.14265996280406255e-02 | -3.40160583148877382e+00 | 9.18443028264240985e-02 | escapes toward k4-side negative G4 |
| 9 10 25 | 2.24437787342958092e-02 | 1.14265996279884607e-02 | 4.79854687825515103e-01 | 3.64071841197107157e-02 | escapes toward k3-side positive G4 |

## Raw-Below Selector Supports

- Parsed raw-below logs: `5`
- All below endpoint by certified upper value: `True`
- All global replay pass: `True`
- Hessian skipped in all logs: `True`
- Minimum endpoint gap: `0.000437882905546775` on `10,13,19`

| support | certified upper | endpoint gap | projected grad | global cert | packaging status |
| :--- | ---: | ---: | ---: | :--- | :--- |
| 10,13,19 | 0.0206265137 | 0.000437882905546775 | 1.976e-07 | PASS (P8 scan; 4096 starts) | raw_below_global_replay_needs_local_hessian_or_interval_packaging |
| 13,22,29 | 0.01851717827 | 0.002547218335546775 | 7.687e-06 | PASS (P8 scan; 4096 starts) | raw_below_global_replay_needs_local_hessian_or_interval_packaging |
| 13,17,24 | 0.01719930671 | 0.003865089895546775 | 9.316e-07 | PASS (P8 scan; 4096 starts) | raw_below_global_replay_needs_local_hessian_or_interval_packaging |
| 13,29,30 | 0.01537241729 | 0.005691979315546775 | 3.196e-06 | PASS (P8 scan; 4096 starts) | raw_below_global_replay_needs_local_hessian_or_interval_packaging |
| 13,22,25 | 0.01495877202 | 0.006105624585546775 | 6.831e-08 | PASS (P8 scan; 4096 starts) | raw_below_global_replay_needs_local_hessian_or_interval_packaging |

## Raw-Below Strip-Boundary Ledger

- Boundary faces parsed: `10` / `10`
- Complete boundary ledger: `True`
- All boundary values below endpoint: `True`
- All interval endpoint gaps positive: `True`
- Minimum boundary gap: `0.0029449843956699945` on `13 22 29 upper`
- Maximum boundary R: `0.01811941220987678`
- Maximum boundary constraint error: `8.498500177894774e-08`

| support | boundary | R | G4 | constraint | endpoint gap | interval gap positive |
| :--- | :--- | ---: | ---: | ---: | ---: | :---: |
| 10 13 19 | lower | 3.30958253698504989e-03 | -5.02801784933144466e-03 | 1.951e-12 | 1.77548140685617264e-02 | True |
| 10 13 19 | upper | 9.45494429559429195e-03 | 1.14265996278651409e-02 | 1.369e-13 | 1.16094523099524831e-02 | True |
| 13 17 24 | lower | 1.05274278040481985e-02 | -5.02801778462686443e-03 | 6.666e-11 | 1.05369688014985766e-02 | True |
| 13 17 24 | upper | 1.92090859191442202e-03 | 1.14265955031399424e-02 | 4.125e-09 | 1.91434880136323524e-02 | True |
| 13 22 25 | lower | 1.20125698818140392e-02 | -5.02801785099156462e-03 | 2.909e-13 | 9.05182672373273586e-03 | True |
| 13 22 25 | upper | 1.45340720621077567e-02 | 1.14265996280873763e-02 | 8.530e-14 | 6.53032454343901829e-03 | True |
| 13 22 29 | lower | 1.61843308984063253e-02 | -5.02793286628069533e-03 | 8.499e-08 | 4.88006570714044977e-03 | True |
| 13 22 29 | upper | 1.81194122098767806e-02 | 1.14265996279395744e-02 | 6.251e-14 | 2.94498439566999448e-03 | True |
| 13 29 30 | lower | 1.11590888506070077e-02 | -5.02801784737400177e-03 | 3.908e-12 | 9.90530775493976730e-03 | True |
| 13 29 30 | upper | 1.41447437231301265e-02 | 1.14265995910406044e-02 | 3.696e-11 | 6.91965288241664853e-03 | True |

## Raw-Below Interior Quotient Diagnostic

- Summary JSON: `paper3\results\20260613_021357_certify_k34_raw_below_interior_quotient_k34_raw_below_interior_quotient_v8_abs_satellite_tube_cuda\20260613_021357_raw_below_interior_quotient_summary.json`
- Active quotient Hessian strictly negative in all cases: `False`
- Inactive log-amplitude KKT sign nonpositive in all cases: `True`
- Minimum certified endpoint gap: `0.00043788290554677353`
- Maximum quotient Hessian eigenvalue: `1.2166449216153287e-07`
- Maximum satellite X fraction after keeping three core modes: `0.0001319602898723904`
- Maximum observed positive satellite correction: `0.0`
- One-satellite channel vanishes in all cases: `True`
- Maximum two-plus satellite absolute contribution at stored point: `7.808786842320888e-06`
- Maximum two-plus satellite coarse Cauchy coefficient sum: `114.93279695872641`
- Satellite release Hessian of `-R` positive in all cases: `True`
- Minimum satellite release Hessian eigenvalue of `-R`: `7.329651480118961e-08`
- Minimum crude cubic half-gap tube delta: `0.000398305417031442`
- Maximum observed fraction of that tube: `0.11211949138311406`
- Minimum absolute two-plus half-gap tube delta: `5.0595518674385376e-05`
- Maximum observed fraction of the absolute two-plus tube: `0.6190826003129184`

| support | active/total modes | endpoint gap | core gap | satellite X frac | cubic sat/tube | abs 2+ sat/tube | sat Hessian min eig | max quotient Hessian eig |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10,13,19 | 13/36 | 4.37882905546773532e-04 | 4.37778465045706583e-04 | 3.369e-06 | 8.459e-03 | 6.659e-02 | 5.733e-07 | 6.274e-11 |
| 13,17,24 | 38/48 | 3.86508989554677426e-03 | 3.86502647080125272e-03 | 1.320e-04 | 1.121e-01 | 6.191e-01 | 1.890e-07 | 1.217e-07 |
| 13,22,25 | 22/36 | 6.10562458554677473e-03 | 6.10546945768389131e-03 | 6.906e-06 | 2.749e-03 | 9.692e-03 | 1.297e-06 | 3.291e-10 |
| 13,22,29 | 31/60 | 2.54721833554677479e-03 | 2.54647934424049455e-03 | 3.813e-05 | 4.300e-02 | 2.692e-01 | 7.330e-08 | 2.813e-11 |
| 13,29,30 | 41/72 | 5.69197931554677426e-03 | 5.69185346641306288e-03 | 6.196e-05 | 6.823e-02 | 2.720e-01 | 7.951e-08 | 1.021e-08 |

## Raw-Below Exterior-Of-Tube Screen

- Summary JSON: `paper3\results\20260613_021837_screen_k34_selector_strip_k34_selector_strip_raw_below_exterior_tube_v1\20260613_021837_selector_strip_summary.json`
- Satellite exterior only: `True`
- All supports excluded numerically: `True`

| support | best exterior R | G4 | endpoint gap | satellite delta | tube delta |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 10 13 19 | 8.08825146422223473e-03 | 1.11913715396835516e-02 | 1.29761451413245403e-02 | 3.699e-01 | 5.060e-05 |
| 13 17 24 | 8.70162108171814754e-03 | -1.04567895068787865e-03 | 1.23627755238286275e-02 | 5.028e-04 | 2.132e-04 |
| 13 22 25 | 1.39942580319309187e-02 | 7.31652338960817484e-03 | 7.07013857361585632e-03 | 1.000e+00 | 7.126e-04 |
| 13 22 29 | 1.81202559067884933e-02 | 1.14058775156025798e-02 | 2.94414069875828174e-03 | 1.000e+00 | 1.416e-04 |
| 13 29 30 | 1.36203066525299576e-02 | 7.31666497020369132e-03 | 7.44408995301681746e-03 | 1.000e+00 | 2.278e-04 |

## Face-Class Ledger

| face class | residual band | outside slots | K34 signature | effective triads | supports | status |
| :--- | :--- | ---: | :---: | ---: | ---: | :--- |
| k3_k4_skeleton_bridge_face | no_residual_shell | 0 | no | 17952 | 57 | open_skeleton_bridge_cover |
| k3_k4_skeleton_bridge_face | no_residual_shell | 0 | yes | 4224 | 30 | open_skeleton_bridge_cover |
| window_residual_face | i3_residual_only | 1 | no | 26856 | 75 | outside_k34_signature_rows_support_level_selector_exclusion_still_required |
| window_residual_face | i3_residual_only | 1 | yes | 5544 | 49 | open_single_band_residual_leakage_or_support_certificate |
| window_residual_face | i3_residual_only | 2 | no | 7272 | 32 | outside_k34_signature_rows_support_level_selector_exclusion_still_required |
| window_residual_face | i3_residual_only | 2 | yes | 1512 | 20 | open_single_band_residual_leakage_or_support_certificate |
| window_residual_face | i3_residual_only | 3 | no | 288 | 2 | outside_k34_signature_rows_support_level_selector_exclusion_still_required |
| window_residual_face | i4_residual_only | 1 | no | 25128 | 82 | outside_k34_signature_rows_support_level_selector_exclusion_still_required |
| window_residual_face | i4_residual_only | 1 | yes | 4824 | 50 | open_single_band_residual_leakage_or_support_certificate |
| window_residual_face | i4_residual_only | 2 | no | 7128 | 27 | outside_k34_signature_rows_support_level_selector_exclusion_still_required |
| window_residual_face | i4_residual_only | 2 | yes | 1008 | 11 | open_single_band_residual_leakage_or_support_certificate |
| window_residual_face | i4_residual_only | 3 | no | 432 | 3 | outside_k34_signature_rows_support_level_selector_exclusion_still_required |
| window_residual_face | mixed_i3_i4_residual | 2 | no | 12600 | 44 | outside_k34_signature_rows_support_level_selector_exclusion_still_required |
| window_residual_face | mixed_i3_i4_residual | 2 | yes | 2664 | 26 | open_mixed_residual_leakage_or_involution_classification |
| window_residual_face | mixed_i3_i4_residual | 3 | no | 3456 | 14 | outside_k34_signature_rows_support_level_selector_exclusion_still_required |
| window_residual_face | mixed_i3_i4_residual | 3 | yes | 432 | 5 | open_mixed_residual_leakage_or_involution_classification |
| k3_endpoint_face | no_residual_shell | 0 | no | 840 | 7 | covered_by_U3_endpoint_chart |
| k3_endpoint_face | no_residual_shell | 0 | yes | 240 | 3 | covered_by_U3_endpoint_chart |
| k4_endpoint_safe_face | no_residual_shell | 0 | no | 13392 | 44 | covered_inside_U4_chart_after_U4_radius_is_fixed |
| k4_endpoint_safe_face | no_residual_shell | 0 | yes | 4176 | 28 | covered_inside_U4_chart_after_U4_radius_is_fixed |
