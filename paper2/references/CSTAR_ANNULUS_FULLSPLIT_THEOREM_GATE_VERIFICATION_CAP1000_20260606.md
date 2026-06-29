# C* Full-Split Theorem Gate Verification

This is a fast verifier packet, not manuscript text.  It reads the saved
stabilized full finite shell-split candidate and records the current
proof-gate status without launching a new scan.

## Verdict

The numerical target is stable and should be treated as the active C*
annulus value for proof work.  The equality theorem is not yet paper-ready
because the remaining gates need interval packaging, not more numerical
search.

## Candidate

- source JSON: `scripts\results\cstar_annulus_fullsplit_cap1000_polish_20260606.json`
- ratio: `0.31226426322102058`
- target display value: `0.31226427000000001`
- target minus ratio: `6.7789794333528164e-09`
- previous ratio: `0.31226426322102052`
- gain vs previous: `5.5511151231257827e-17`
- group count: `801`
- active coordinates: `647`
- free coordinates: `647`
- zero-bound coordinates: `154`
- upper-bound coordinates: `0`
- scale cap: `1000`
- max scale: `200.00000000868158`
- coordinates at scale cap: `0`

## KKT Snapshot

- gradient norm: `2.5274837866370627e-10`
- gradient max abs: `6.520921662641337e-11`
- active gradient max abs: `3.9637331638491327e-12`
- free gradient max abs: `3.9637331638491327e-12`
- zero-bound gradient max: `-3.5345695459004496e-13`
- zero-bound gradient min: `-6.520921662641337e-11`
- zero-bound positive gradient count: `0`
- upper-bound gradient max: `not available`
- upper-bound gradient min: `not available`
- upper-bound negative gradient count: `0`

### Largest Active Gradients

| rank | group | scale | gradient |
|---:|---|---:|---:|
| `1` | `r0805_s1574` | `0` | `-6.520921662641337e-11` |
| `2` | `r0824_s1601` | `0` | `-5.6266389390343377e-11` |
| `3` | `r0818_s1586` | `0` | `-5.114679920259292e-11` |
| `4` | `r0792_s1553` | `0` | `-4.6626916333087966e-11` |
| `5` | `r0853_s1629` | `0` | `-4.5168118254409043e-11` |
| `6` | `r0820_s1589` | `0` | `-4.4021014625129092e-11` |
| `7` | `r0806_s1566` | `0` | `-3.9862753048748097e-11` |
| `8` | `r0852_s1625` | `0` | `-3.8774086364558625e-11` |
| `9` | `r0816_s1569` | `0` | `-3.8657211289895577e-11` |
| `10` | `r0867_s1641` | `0` | `-3.8470792612848864e-11` |
| `11` | `r0832_s1595` | `0` | `-3.7753601306481109e-11` |
| `12` | `r0841_s1602` | `0` | `-3.5591377388449202e-11` |

### Largest Scales

| rank | group | scale | gradient |
|---:|---|---:|---:|
| `1` | `r1287_s2141` | `200.00000000868158` | `3.3154786613679889e-13` |
| `2` | `r1283_s2129` | `200.00000000786173` | `3.0023774939491138e-13` |
| `3` | `r1277_s2126` | `200.00000000605803` | `2.3135456161677136e-13` |
| `4` | `r1308_s2150` | `200.00000000602759` | `2.3019188142623887e-13` |
| `5` | `r1289_s2133` | `200.00000000589685` | `2.2519924601724719e-13` |
| `6` | `r1298_s2138` | `200.00000000581056` | `2.2190399258337981e-13` |
| `7` | `r1313_s2162` | `200.00000000514902` | `1.9663939671355861e-13` |
| `8` | `r1321_s2169` | `200.00000000452621` | `1.7285490546061577e-13` |
| `9` | `r1280_s2121` | `200.00000000445439` | `1.701127590621209e-13` |
| `10` | `r1303_s2134` | `200.00000000439769` | `1.679464900463448e-13` |
| `11` | `r1329_s2174` | `200.0000000043498` | `1.6611812799596516e-13` |
| `12` | `r1307_s2142` | `200.00000000419473` | `1.6019597437405335e-13` |

## Hessian Evidence

- Hessian JSON: `scripts\results\cstar_annulus_direct_hessian_fullsplit_cap1000_free_20260606.json`
- Hessian source: `scripts\results\cstar_annulus_fullsplit_cap1000_polish_20260606.json`
- source matches stabilized candidate: `True`
- active only: `False`
- selected count: `647`
- finite-difference step: `0.01`
- min eigenvalue: `-7.2683307159959775e-05`
- max eigenvalue: `7.1307078621873765e-57`
- zero diagonal count: `6`

Effective negative submatrices after dropping numerically zero diagonal rows:

| diag threshold | kept | dropped | max eigenvalue |
|---:|---:|---:|---:|
| `0` | `641` | `6` | `-3.0235227279516019e-23` |
| `1.0000000000000001e-30` | `641` | `6` | `-3.0235227279516019e-23` |
| `9.9999999999999992e-25` | `641` | `6` | `-3.0235227279516019e-23` |
| `9.9999999999999995e-21` | `636` | `11` | `-1.5999236886811338e-20` |
| `1.0000000000000001e-18` | `616` | `31` | `-1.3636744180048693e-18` |
| `9.9999999999999998e-17` | `577` | `70` | `-1.0105301588651645e-16` |
| `1e-14` | `478` | `169` | `-1.045767926727662e-14` |
| `9.9999999999999998e-13` | `191` | `456` | `-1.0064989823141212e-12` |

Curvature-normalized negative Hessian margins:

| diag threshold | kept | dropped | min eig | Gershgorin margin | max offdiag row sum | entry radius target |
|---:|---:|---:|---:|---:|---:|---:|
| `0` | `641` | `6` | `0.69033821039725485` | `0.50538223782086966` | `0.49461776217913034` | `0.00039482987329755441` |
| `1.0000000000000001e-30` | `641` | `6` | `0.69033821039725485` | `0.50538223782086966` | `0.49461776217913034` | `0.00039482987329755441` |
| `9.9999999999999992e-25` | `641` | `6` | `0.69033821039725485` | `0.50538223782086966` | `0.49461776217913034` | `0.00039482987329755441` |
| `9.9999999999999995e-21` | `636` | `11` | `0.69033833783795695` | `0.50541042551502047` | `0.49458957448497953` | `0.00039796096497245707` |
| `1.0000000000000001e-18` | `616` | `31` | `0.69034517372077708` | `0.50609049428470598` | `0.49390950571529402` | `0.00041145568641033007` |
| `9.9999999999999998e-17` | `577` | `70` | `0.69038347888127805` | `0.50852649023004615` | `0.49147350976995385` | `0.00044142924499135952` |
| `1e-14` | `478` | `169` | `0.69052990004257697` | `0.51777064985618759` | `0.48222935014381241` | `0.00054273653024757614` |
| `9.9999999999999998e-13` | `191` | `456` | `0.69076287303148509` | `0.55304203925149298` | `0.44695796074850702` | `0.0014553737875039289` |

## Cutoff-Zero Row Audit

The cutoff-zero lemma is only applied to the final active rows whose
Hessian diagonal vanishes at threshold `1e-30`.  It is not applied
to the larger numerical tail.

- cutoff threshold: `1.0000000000000001e-30`
- cutoff rows: `6`
- exact zero direction rows: `3`
- total one-high triads: `0`
- total shared one-high triads: `0`
- total linear abs: `5.5965044354606178e-49`
- total delta-X abs: `8.1022798828489255e-50`
- total delta-D abs: `2.8527750525380219e-48`
- max direction abs: `1.1504218008741535e-24`
- structural zero pass: `True`

| selected | coordinate | group | one-high triads | direction max abs | ledger linear abs |
|---:|---:|---|---:|---:|---:|
| `641` | `795` | `r1408_s2260` | `0` | `9.9078875667401822e-25` | `2.9956936011879278e-49` |
| `642` | `796` | `r1409_s2256` | `0` | `7.7601843534171309e-25` | `1.4932461352880348e-49` |
| `643` | `797` | `r1410_s2240` | `0` | `1.1504218008741535e-24` | `1.1075646989846555e-49` |
| `644` | `798` | `r1411_s2257` | `0` | `0` | `0` |
| `645` | `799` | `r1412_s2259` | `0` | `0` | `0` |
| `646` | `800` | `r1413_s2253` | `0` | `0` | `0` |

## Active Quadratic Residual

This table uses the transferred active Hessian and the final
stabilized gradient to estimate the residual gain left on each
effective negative submatrix.  Rows dropped by the diagonal
threshold still require cutoff-zero row handling; this table is
a proof-target diagnostic, not a standalone theorem.  The feasible
gain column is a line-search diagnostic, not a rigorous upper bound
on the active KKT residual.

| diag threshold | kept | dropped | kept grad max | dropped grad max | Newton gain | feasible alpha | feasible gain |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `0` | `641` | `6` | `3.9637331638491327e-12` | `2.7245087827817748e-31` | `1.8342288867546334e-09` | `0.019126827393006307` | `6.9494932542630186e-11` |
| `1.0000000000000001e-30` | `641` | `6` | `3.9637331638491327e-12` | `2.7245087827817748e-31` | `1.8342288867546334e-09` | `0.019126827393006307` | `6.9494932542630186e-11` |
| `9.9999999999999992e-25` | `641` | `6` | `3.9637331638491327e-12` | `2.7245087827817748e-31` | `1.8342288867546334e-09` | `0.019126827393006307` | `6.9494932542630186e-11` |
| `9.9999999999999995e-21` | `636` | `11` | `3.9637331638491327e-12` | `1.3545036871536533e-16` | `1.8308728430152296e-09` | `0.043087019332165841` | `1.5437470777813931e-10` |
| `1.0000000000000001e-18` | `616` | `31` | `3.9637331638491327e-12` | `5.1567240895401785e-15` | `1.6652810138151574e-09` | `0.15229869432465548` | `4.6861423442236214e-10` |
| `9.9999999999999998e-17` | `577` | `70` | `3.9637331638491327e-12` | `6.7410368154134784e-14` | `8.317454617604463e-10` | `0.80397443328745388` | `7.9978479167890506e-10` |
| `1e-14` | `478` | `169` | `3.9637331638491327e-12` | `3.3154786613679889e-13` | `1.1595484481510852e-18` | `1` | `1.1595484481510852e-18` |
| `9.9999999999999998e-13` | `191` | `456` | `3.9637331638491327e-12` | `3.3154786613679889e-13` | `1.1571724929806799e-18` | `1` | `1.1571724929806799e-18` |

## Hessian Transfer

No Hessian-transfer JSON supplied.

## Interval And Rounding Gate Ledger

This section converts the remaining theorem packaging work into
explicit numerical thresholds.  It is not a new search and it does
not rely on any exploratory rank scan.

- target margin to `0.31226425`: `6.7789793999999999e-09`
- relative target margin: `2.1709110514519041e-08`
- group-normalization denominator: `1.0017749498175816`
- implied normalized numerator: `0.31281851661806204`
- denominator gamma count: `4838`
- denominator gamma radius: `5.3712589931393922e-13`
- numerator relative gamma budget after denominator: `2.1708573388619728e-08`
- max numerator gamma operation count for display target: `195533441`

### Replay Gamma Budget

| numerator operation count | gamma | ratio radius | target slack after radius | pass |
|---:|---:|---:|---:|---:|
| `100000` | `1.1102230246374825e-11` | `3.6345549712005601e-12` | `6.7753448450287994e-09` | `True` |
| `500000` | `5.5511151234339314e-11` | `1.7501873963947808e-11` | `6.7614775260360525e-09` | `True` |
| `1000000` | `1.110223024748416e-10` | `3.4836022706613904e-11` | `6.744143377293386e-09` | `True` |
| `2000000` | `2.2204460497433512e-10` | `6.9504320197719513e-11` | `6.7094750798022801e-09` | `True` |
| `5000000` | `5.5511151262072704e-10` | `1.7350921271722382e-10` | `6.605470187282776e-09` | `True` |
| `7000000` | `7.7715611784158119e-10` | `2.4284580776871622e-10` | `6.5361335922312836e-09` | `True` |
| `10000000` | `1.1102230258577516e-09` | `3.4685070040368912e-10` | `6.4321286995963108e-09` | `True` |

### KKT Interval Targets

- zero-bound interval radius required for strict complementarity: `3.5345695459004496e-13`
- half-slack working target for zero-bound gradients: `1.7672847729502248e-13`
- active gradient max abs to absorb through quadratic residual: `3.9637331638491327e-12`

### Active Residual Budget

| diag threshold | kept | dropped | feasible gain | target slack after gain | max gamma count after gain | pass before rounding |
|---:|---:|---:|---:|---:|---:|---:|
| `0` | `641` | `6` | `6.9494932542630186e-11` | `6.7094844674573702e-09` | `193528874` | `True` |
| `1.0000000000000001e-30` | `641` | `6` | `6.9494932542630186e-11` | `6.7094844674573702e-09` | `193528874` | `True` |
| `9.9999999999999992e-25` | `641` | `6` | `6.9494932542630186e-11` | `6.7094844674573702e-09` | `193528874` | `True` |
| `9.9999999999999995e-21` | `636` | `11` | `1.5437470777813931e-10` | `6.6246046922218605e-09` | `191080534` | `True` |
| `1.0000000000000001e-18` | `616` | `31` | `4.6861423442236214e-10` | `6.3103651655776379e-09` | `182016359` | `True` |
| `9.9999999999999998e-17` | `577` | `70` | `7.9978479167890506e-10` | `5.9791946083210946e-09` | `172463812` | `True` |
| `1e-14` | `478` | `169` | `1.1595484481510852e-18` | `6.7789793988404515e-09` | `195533441` | `True` |
| `9.9999999999999998e-13` | `191` | `456` | `1.1571724929806799e-18` | `6.7789793988428272e-09` | `195533441` | `True` |

The tightest displayed residual-plus-target row is:

- threshold: `9.9999999999999998e-17`
- kept/dropped: `577/70`
- remaining target slack after feasible quadratic gain: `5.9791946083210946e-09`
- max numerator gamma count after that gain: `172463812`

### Hessian Interval Targets

For a kept active block, an entrywise Hessian interval radius `rho`
has spectral perturbation at most `n rho`.  The table below gives
the strict radius needed to preserve negative curvature by that
simple bound.

| diag threshold | kept | dropped | spectral margin | entrywise radius target | half-margin target |
|---:|---:|---:|---:|---:|---:|
| `0` | `641` | `6` | `3.0235227279516019e-23` | `4.7168841309697378e-26` | `2.3584420654848689e-26` |
| `1.0000000000000001e-30` | `641` | `6` | `3.0235227279516019e-23` | `4.7168841309697378e-26` | `2.3584420654848689e-26` |
| `9.9999999999999992e-25` | `641` | `6` | `3.0235227279516019e-23` | `4.7168841309697378e-26` | `2.3584420654848689e-26` |
| `9.9999999999999995e-21` | `636` | `11` | `1.5999236886811338e-20` | `2.5156032840898331e-23` | `1.2578016420449166e-23` |
| `1.0000000000000001e-18` | `616` | `31` | `1.3636744180048693e-18` | `2.2137571720858267e-21` | `1.1068785860429133e-21` |
| `9.9999999999999998e-17` | `577` | `70` | `1.0105301588651645e-16` | `1.7513520950869403e-19` | `8.7567604754347014e-20` |
| `1e-14` | `478` | `169` | `1.045767926727662e-14` | `2.1877990098905062e-17` | `1.0938995049452531e-17` |
| `9.9999999999999998e-13` | `191` | `456` | `1.0064989823141212e-12` | `5.2696281796550847e-15` | `2.6348140898275424e-15` |

## Gate Table

| gate | status | evidence | paper ready |
|---|---|---|---:|
| stabilized finite candidate | closed numerically | scale-max stress moved the value by < 2e-10 and hit no cap | `True` |
| zero-bound KKT complementarity | closed in floating replay | 154 zero-bound coordinates, positive zero gradients = 0 | `False` |
| upper-bound KKT complementarity | not active | 0 upper-bound coordinates, negative upper gradients = 0 | `False` |
| active KKT stationarity | polished but not interval closed | free gradient max abs = 3.964e-12 | `False` |
| active Hessian/Schur | normalized margin quantified | Hessian source matches candidate | `False` |
| cutoff-zero rows | closed structurally | six no-triad rows audited at the active Hessian cutoff | `True` |
| finite replay interval | threshold quantified | display target survives gamma_1e6 replay = True; outward-rounded replay still required | `False` |
| far-tail interface | open packaging | finite one-high table ends at |k|^2=2260; existing far-tail theorem must be attached | `False` |

## Next Proof Action

Do not run another broad rank-window scan by default.  The next useful
implementation is now narrow: run or write the outward-rounded replay
against the thresholds above, write the normalized Hessian row-sum
certificate, close the active KKT residual, and attach the far-tail
theorem.  The six cutoff-zero rows are already audited as no-triad
structural zeros in this packet.
