# Principle-11 Cancellation Analysis for k=8–12

**Companion to:** `references/SPECTRAL_TAXONOMY_TAIL_REPAIR.md`
**Script:** `scripts/p11_cancellation.py`
**Status:** The joint Schur and weighted-row finite certificates close k=11 and
k=12.  The midpoint-centered row certificate closes k=8, k=9, and k=10 as upper
bounds below $C(I_2)$; see §5.3 and §6.  The exact optimizer values for k=8--10
remain uncertified and are not used as proof inputs.

---

## CRITICAL: Read this first if you are resuming this work

This document was written specifically to prevent terminology confusion.
A previous analysis session applied the P11 cancellation formula incorrectly and
drew false conclusions. **Section 5 explicitly records the error.**

---

## 1. Terminology and Notation Reference

Every symbol used in this document is defined here. Never substitute meanings.

### 1.1 The primary constant

$$C(I_k) = \sup_{\substack{v \in L^2(\mathbb{T}^3),\ \nabla \cdot v = 0 \\ \operatorname{supp}\hat{v} \subset I_k}} R(v)$$

where

$$R(v) = \frac{B(v,v,\Delta v)}{\|\nabla v\|^2_{L^2}\,\|\Delta v\|_{L^2}},
\quad I_k = \bigl\{n \in \mathbb{Z}^3 : 2^k \le |n|^2 < 2^{k+1}\bigr\}.$$

- **Units:** dimensionless.
- **Scaling:** $C(I_k) \sim 2^{-k/2}$ for large $k$ (decays to zero).
- **Goal:** prove $C(I_k) < C(I_2) = 0.022741865409341\ldots$ for all $k \ge 8$.

### 1.2 The rescaled constant

$$C_{\mathrm{res}}(k) := C(I_k) \times 2^{k/2}.$$

- **Units:** dimensionless.
- **Scaling:** converges to a finite limit for large $k$ (approximately constant once the
  exceptional blocks k=6–12 are passed).
- $C_{\mathrm{res}}(k)$ appears naturally in the Lions-paired kernel analysis because
  rescaling maps $I_k$ to the fixed annulus $S = \{1 \le |\xi|^2 < 2\}$.
- **Do NOT confuse** $C(I_k)$ with $C_{\mathrm{res}}(k)$.

**Conversion table:**

| $k$ | $C(I_k)$ | $C_{\mathrm{res}}(k)$ | Status |
|-----|----------|----------------------|--------|
| 1 | 0 | 0 | Exact |
| 2 | 0.022741865409341... | 0.045483730819... | Exact |
| 3 | <= 0.021936470 | <= 0.062045706 | Conservative comparison bound; refined algebraic/nucleus candidate is 0.021936469459403... |
| 4 | 0.021064396605546... | 0.084258... | GPU cert, 50 dps |
| 5 | 0.018479317637642... | 0.104535... | GPU cert, 50 dps |
| 6 | 0.020443793141444... | 0.163550... | mpmath + Hessian cert |
| 7 | 0.020280600900469... | 0.229449... | GPU cert May 2026 |
| 8 | < 0.021262659 | <= 0.340202543023 | Midpoint-centered mixed certificate |
| 9 | < 0.016346813 | <= 0.369886147671 | Midpoint-centered mixed certificate |
| 10 | < 0.011701242 | <= 0.374439743029 | Midpoint-centered mixed certificate |
| 11 | < 0.021327938 | <= 0.965192260185 | Weighted-row joint Schur certificate |
| 12 | < 0.020366627 | <= 1.303464115585 | Joint Schur finite-cell certificate |
| ≥13 | < C(I₂) | < C(I₂)·2^{k/2} | PROVED: tail_cert.py |

The k=3 closed-form reduction is valuable structural evidence, but it is not yet
an exact global theorem for `C(I_3)`; see `references/K3_CLOSED_FORM.md`.  The
paper-facing statement is the conservative comparison bound
`C(I_3) <= 0.021936470 < C(I_2)`, with the refined algebraic/nucleus candidate
recorded separately.

> **⚠ NEVER USE** stale restricted-run values: k=8: 0.008110, k=9: 0.006520, k=10: 0.004165.
> These came from anchor-restricted (not full-block) scans. They are wrong.
>
> **⚠ The k=8 Adam incumbent (~0.020629) is NOT a certified lower bound.** It is a
> float-precision optimizer output that has not been mpmath-verified, Hessian-certified,
> or globally scanned. The true optimizer may differ. The block is now closed by
> an analytic upper bound, not by direct optimizer certification.

### 1.3 The spectral-taxonomy relay tower bound

$$T_{\mathrm{relay}}(k) = \frac{1.4601017200\ldots}{2^{k/2}}$$

- This is an **upper bound on $C(I_k)$**, proved by Lions-paired kernel analysis
  in `SPECTRAL_TAXONOMY_TAIL_REPAIR.md`.
- The constant $1.4601 = C_{000,\mathrm{pair}}^* + R_{\mathrm{ord}}
  = 0.5732817578 + 0.8868199623$ is a bound on $C_{\mathrm{res}}(k)$ for large $k$.
- $T_{\mathrm{relay}}(k)$ bounds the **net signed value** $B(v,v,\Delta v)/(X^2 D)$,
  NOT the triangle sum (see §2.3).
- For $k \ge 13$: the finite-$k$ Riemann-sum certificate (script: `tail_cert.py`)
  gives $U(13) = 1.9105778717 < 2.0583586877 = C(I_2) \cdot 2^{13/2}$.
  This proves $C(I_k) < C(I_2)$ for all $k \ge 13$.

**Numerical values of** $T_{\mathrm{relay}}(k)$:

| $k$ | $T_{\mathrm{relay}}(k)$ | $T_{\mathrm{relay}}/C(I_2)$ |
|-----|------------------------|----------------------------|
| 8 | 0.09125636 | 4.011× |
| 9 | 0.06452799 | 2.836× |
| 10 | 0.04562818 | 2.006× |
| 11 | 0.03226399 | 1.418× |
| 12 | 0.02281409 | 1.003× |
| 13 | 0.01613200 | 0.709× |

The relay tower alone closes $k \ge 13$ but **fails for $k = 8$–$12$** because
$T_{\mathrm{relay}}(k) > C(I_2)$ in that range.

### 1.4 Triad-level quantities

For a specific field $u$ on $I_k$, write the NS trilinear form as a sum over
all resonant ordered triples $(\ell, r, s)$ with $\ell = r + s$:

$$B(u,u,\Delta u) = \sum_t B_t(u), \quad
B_t(u) = -|\ell|^2\,\operatorname{Im}\bigl[(\hat{u}_s \cdot \boldsymbol{s})\,
(\hat{u}_\ell^* \cdot \hat{u}_r)\bigr].$$

Each $B_t$ is real-valued and carries a sign.

**Implementation guardrail (May 2026):** the only triad class currently safe to
delete at cache-build time is the same-shell class
$|\ell|^2=|r|^2=|s|^2$.  The broader filter $|\ell|^2=|r|^2$ is invalid
for the ordered numerator above: the ULU subclass can have nonzero aggregate
contribution.  `scripts/gap3/multi_mode_beta_bound.py` now records this as
`TRIAD_FILTER_VERSION = 2` and rebuilds caches whose metadata or array lengths
do not match the corrected rule.  Any diagnostic table generated from an older
version-0 cache should be treated as non-proof data unless the streamed replay
matches the checkpoint value.

**Positive and negative parts:**

$$B_{\mathrm{pos}}(u) = \sum_{B_t > 0} B_t(u) \ge 0, \quad
B_{\mathrm{neg}}(u) = \sum_{B_t < 0} B_t(u) \le 0.$$

$$B_{\mathrm{net}}(u) = B_{\mathrm{pos}}(u) + B_{\mathrm{neg}}(u)
= B_{\mathrm{pos}}(u) - |B_{\mathrm{neg}}(u)|.$$

**Triangle sum (field-specific):**

$$T_{\mathrm{field}}(u) = \frac{B_{\mathrm{pos}}(u) + |B_{\mathrm{neg}}(u)|}{X(u)^2 D(u)}
= \frac{\sum_t |B_t(u)|}{X(u)^2 D(u)}.$$

This is the **per-triad triangle bound** at the specific field $u$.
$T_{\mathrm{field}}(u) \ge |B_{\mathrm{net}}(u)|/(X^2 D)$ always.

> **Critical distinction:** $T_{\mathrm{relay}}(k)$ bounds $C(I_k) = \sup_u B_{\mathrm{net}}/(X^2 D)$.
> It does **NOT** bound $\sup_u T_{\mathrm{field}}(u)$.
> In fact $\sup_u T_{\mathrm{field}}(u) = \infty$ (fields with large triangle sum and
> near-perfect cancellation are not excluded by the relay tower analysis).

### 1.5 The Principle-11 cancellation ratio

$$r(u) = \frac{|B_{\mathrm{neg}}(u)|}{B_{\mathrm{pos}}(u)} \in [0, 1).$$

- $r = 0$: all triad contributions are positive (no inter-triad cancellation).
- $r = 1$: perfect cancellation ($B_{\mathrm{net}} = 0$).
- $r$ is **scale-invariant** (it is a ratio; multiplying $u$ by a constant changes
  neither numerator nor denominator).

**Effective coupling fraction:**

$$\mathrm{eff}(u) = \frac{1-r(u)}{1+r(u)} \in (0,1].$$

The formula connecting $T_{\mathrm{field}}$, $r$, and $B_{\mathrm{net}}$ is:

$$\frac{B_{\mathrm{net}}(u)}{X(u)^2 D(u)} = T_{\mathrm{field}}(u) \times \mathrm{eff}(u).
\tag{P11-id}$$

This is an **exact identity**, not an inequality. It holds at every field $u$.

> ⚠ **Formula is NOT** $(1-r)$. It is $(1-r)/(1+r)$. An old version of the analysis
> script used $(1-r)$; all computations here use the correct ratio.

**Extremal value:**

$$r_{\mathrm{extremal}}(k) = r(u_{\mathrm{opt}}),$$

where $u_{\mathrm{opt}}$ is the **global maximizer** of $R(u)$ on fields supported in $I_k$.
This is the minimum of $r$ over all fields that achieve (or approach) $C(I_k)$.

---

## 2. The Exact P11 Identity and Its Meaning

### 2.1 At the certified optimizer

At the global optimizer $u_{\mathrm{opt}}$ of $R$ on $I_k$:

$$C(I_k) = R(u_{\mathrm{opt}}) = T_{\mathrm{field}}(u_{\mathrm{opt}}) \times \mathrm{eff}(k).$$

This is an **exact relation** between three computable quantities:

| Quantity | Meaning | Computed from |
|----------|---------|---------------|
| $C(I_k)$ | The certified supremum | GPU certification |
| $T_{\mathrm{field}}(u_{\mathrm{opt}})$ | Triangle sum at optimizer / ($X^2 D$) | warm-state + triad enumeration |
| $\mathrm{eff}(k) = (1-r)/(1+r)$ | Effective coupling fraction at optimizer | same |

### 2.2 Certified data (k=4, k=5)

The warm states `results/warm_state/k4_warm.npz` and `results/warm_state/k5_warm.npz`
ARE the certified global optima (full-block, GPU-certified to 50 decimal places).
So $r_{\mathrm{obs}}(4) = r_{\mathrm{extremal}}(4)$ and similarly for $k=5$.

From `p11_cancellation.py` (verified output):

```
k=4  (certified global opt):
  C(I_4)           = 0.021064396605547
  T_relay(4)       = 0.36502543  (Lions-paired bound on C(I_4))
  T_field(u_opt)   = 0.11019708  (actual triangle sum / X^2 D at optimizer)
  r_extremal(4)    = 0.67896...
  eff(4)           = 0.19115...
  VERIFY: T_field × eff = 0.02106440  ✓ (matches C(I_4) exactly)
  Ratio T_relay / T_field = 0.36503 / 0.11020 = 3.31×  (relay tower is loose)

k=5  (certified global opt):
  C(I_5)           = 0.018479317637642
  T_relay(5)       = 0.25811196
  T_field(u_opt)   = 0.10629251
  r_extremal(5)    = 0.70384...
  eff(5)           = 0.17385...
  VERIFY: T_field × eff = 0.01847932  ✓
  Ratio T_relay / T_field = 0.25811 / 0.10629 = 2.43×
```

**Interpretation:** The relay tower ($T_{\mathrm{relay}}$) overestimates the actual triangle
sum at the optimizer by factors of 3.3× (k=4) and 2.4× (k=5). This looseness
is separate from, and additional to, the inter-triad cancellation captured by $r$.

### 2.3 Why the relay tower does NOT bound $T_{\mathrm{field}}$

The Lions-paired analysis in `SPECTRAL_TAXONOMY_TAIL_REPAIR.md` computes:

$$C_{000,\mathrm{pair}}^* + R_{\mathrm{ord}} = 0.5732817578 + 0.8868199623 = 1.4601017200.$$

This bounds $C_{\mathrm{res}}(k) = C(I_k) \cdot 2^{k/2}$, hence bounds the **net signed**
quantity $B_{\mathrm{net}}/(X^2 D)$. The Lions pairing (via the factors $s - \rho$ and
$t - s$ in the paired kernels $M_{\mathrm{oa}}$ and $M_{\mathrm{pmid}}$) accounts for
the cancellation between paired slot orderings of each triad.

$T_{\mathrm{field}}(u) = \sum_t |B_t|/(X^2 D)$ is the sum of **absolute values** of
all per-triad contributions. It is NOT the net B, and it can exceed $T_{\mathrm{relay}}$.

In fact:

$$\sup_u T_{\mathrm{field}}(u) = +\infty.$$

(Take any field with $B_{\mathrm{net}} = 0$ and amplify arbitrarily — the ratio
$T_{\mathrm{field}}$ is scale-invariant and can be made arbitrarily large along a
sequence of fields with increasing but cancelled positive and negative parts.)

Therefore: **$T_{\mathrm{relay}}(k)$ does NOT bound $T_{\mathrm{field}}$.**

---

## 3. The Formula Error: What Was Computed vs. What Was Claimed

### 3.1 The incorrect claim from the previous session

The previous analysis (summarised in session memory) claimed:

> *Principle-11 bound: $C(I_k) \le T_{\mathrm{relay}}(k) \times (1-r_{\mathrm{ext}})/(1+r_{\mathrm{ext}})$.*

It was further claimed that this closes $k=8$–$12$ using $r_{\mathrm{ext}} \approx 0.679$
(from the k=4 certified optimizer).

**This claim is incorrect.** The formula is not a valid upper bound on $C(I_k)$.

### 3.2 Why the formula is not proved valid (and likely fails for k≥6)

The formula would be a valid upper bound if and only if:

$$T_{\mathrm{relay}}(k) \ge T_{\mathrm{field}}(u_{\mathrm{opt},k}).$$

Using the P11 identity $C(I_k) = T_{\mathrm{field}} \times \mathrm{eff}$, this is equivalent to:

$$T_{\mathrm{relay}}(k) \ge \frac{C(I_k)}{\mathrm{eff}(k)}.$$

**This condition is not established for any $k \ge 6$.** For the two certified blocks:

| $k$ | $T_{\mathrm{relay}}(k)$ | $T_{\mathrm{field}}(u_{\mathrm{opt}})$ | $T_{\mathrm{relay}}/T_{\mathrm{field}}$ | Condition holds? |
|-----|------------------------|---------------------------------------|----------------------------------------|------------------|
| 4 | 0.3650 | 0.1102 | 3.31× | YES |
| 5 | 0.2581 | 0.1063 | 2.43× | YES |

The ratio is **decreasing**. In rescaled units, $\widetilde{T}_{\mathrm{field}}(k) = T_{\mathrm{field}} \cdot 2^{k/2}$
grows from 0.441 (k=4) to 0.601 (k=5), while the relay tower stays fixed at
$1.4601$ in rescaled units. If this growth continues, the condition will eventually fail.

We do not have certified data beyond k=5 to evaluate the condition for k=8–12.
The k=8 block has not been through the full global certification (mpmath verification,
Hessian certification, global scan). **The formula is unproved for k=8–12, and the
trend strongly suggests it fails.** But this is not yet a formal refutation.

### 3.3 What went wrong in the previous session

The previous analysis assumed that because $T_{\mathrm{relay}}$ is an upper bound on
$C(I_k)$, multiplying it by $\mathrm{eff} < 1$ would give a TIGHTER upper bound.
This reasoning is wrong: it is only valid if $T_{\mathrm{relay}} \ge T_{\mathrm{field}}$
at the optimizer, which is a separate and unproved condition.

For $k=4, 5$ the condition happens to hold (verified numerically using the certified
warm states). The analysis extrapolated that it would hold for $k=8$–$12$ using the
observed $r \approx 0.679$ from k=4 as a proxy. This extrapolation is unjustified:

1. The k=8–12 optimizers have NOT been certified; their true $r_{\mathrm{extremal}}$
   values are unknown.
2. The k=8 block has a diffuse mixed A/B/C/D web structure (see §4.2), unlike the
   concentrated top-cluster/bottom-cluster structure of k=4.
3. The ratio $T_{\mathrm{relay}}/T_{\mathrm{field}}$ is decreasing with k (3.31× at k=4,
   2.43× at k=5), so extrapolating that it stays $>1$ at k=8 is not warranted.

### 3.4 The role of the k=8 Adam incumbent

The Adam optimizer for k=8 found an incumbent value of approximately 0.020629
(float32/64 precision). This is **not** a rigorous lower bound on $C(I_8)$ because:

- The evaluation of $R(v)$ at the Adam checkpoint has floating-point error.
- The checkpoint has not been independently verified via mpmath to the required precision.
- The run has not completed the full global certification (Hessian certificate, global scan).

After full global certification, $C(I_8)$ could be higher (if the Adam run hasn't
converged to the global maximum), or it could turn out that additional cancellation
structure moves the certified value to a different regime. **We simply do not know
until the full global cert is completed.**

The Adam incumbent should be treated as a **warm-start target** (a good starting point
for global certification), not as a certified floor.

---

## 4. What the P11 Analysis DOES Correctly Establish

Despite the error in the bound, the analysis produced genuine structural insight.

### 4.1 The output-shell mechanism at k=4 (structural, not accidental)

At the certified k=4 global optimizer, the output-shell (ell-shell) breakdown shows:

```
Top shells (ell²=25,26,29): r_shell = 0.01–0.07  (nearly all positive)
Bottom shells (ell²=16,17,18): r_shell = 12–94   (overwhelmingly negative)
```

**Interpretation:** The optimizer places energy in the "top cluster" of shells
(near $|n|^2 = 2^{k+1} - 1$, e.g. ell²=25,26,29 for k=4). By the geometry of
resonant triads, the output into the "bottom cluster" (near $|n|^2 = 2^k$, e.g.
ell²=16,17,18) is geometrically forced to be negative. This is the content of
**Principle 11: extremal cancellation ratios are structural minima**.

This is directly related to `thm:crystal_evenk` in `paper2/ns_cancellation.tex`:
for even $k$, the bottom shell ($|n|^2 = 2^k$) has 6 axis-aligned modes, and
the B-type crystal structure forces specific coupling signs with the top cluster.

### 4.2 The structure differs between k=4 and k=8

The k=8 block has a **diffuse mixed web** (confirmed by the May 29 checkpoint,
current-filter replay, and SPECTRAL_TAXONOMY §"k=8 guardrail checkpoint"):

```
A: 14.05%, B: 25.40%, C: 22.26%, D: 0.51%  (at 0.5% energy threshold)
```

No single family dominates. The optimizer for k=8 finds a different structural
type than k=4, and the top-cluster / bottom-cluster mechanism is less pronounced.
The corrected May 29 replay gives a high cancellation ratio at the incumbent,
but also a much larger field triangle sum:

```text
r = |negative|/positive = 0.744851262644
eff = (1-r)/(1+r) = 0.146229505528
T_field = sum |B_t|/(X^2D) = 0.1410696172829276
T_field * eff = 0.02062854038024914
rescaled T_field = 2.257113876527
rescaled signed ratio = 0.330056646084
```

This is not certified global-optimum data, but it is a trustworthy replay of the
current Adam checkpoint.  It reinforces the main correction: the P11 cancellation
ratio is not the bottleneck by itself; the triangle field at k=8 is far above the
relay tower, so multiplying a relay bound by `eff` is still invalid.

### 4.3 The rescaled T_field trend

The rescaled field-specific triangle bound at the certified optimizer:

$$\widetilde{T}_{\mathrm{field}}(k) := T_{\mathrm{field}}(u_{\mathrm{opt}}) \times 2^{k/2}$$

| $k$ | $T_{\mathrm{field}}(u_{\mathrm{opt}})$ | $\widetilde{T}_{\mathrm{field}}(k)$ | $C_{000,\mathrm{pair}}^*$ |
|-----|----------------------------------------|--------------------------------------|---------------------------|
| 4 | 0.11020 | 0.4408 | 0.5733 |
| 5 | 0.10629 | 0.6012 | 0.5733 |
| 8 incumbent | 0.14107 | 2.2571 | 0.5733 |

The rescaled $T_{\mathrm{field}}$ at k=4 is BELOW $C_{000,\mathrm{pair}}^*$, but at k=5
it EXCEEDS it. This is consistent with $C_{000,\mathrm{pair}}^*$ bounding the RESCALED
NET $B$ (not the triangle sum), via Lions pairing absorbing part of the
inter-triad cancellation. The k=8 incumbent makes this distinction impossible to
miss: its field triangle sum is almost four times the primitive paired constant,
while the signed ratio is still below the k=2 threshold.

### 4.4 Correct summary of P11 findings

| Claim | Status |
|-------|--------|
| $C(I_4) = T_{\mathrm{field}}(u_{\mathrm{opt}}) \times \mathrm{eff}(4)$ | EXACT (by definition) |
| $r_{\mathrm{extremal}}(4) = 0.6790$, $\mathrm{eff}(4) = 0.1912$ | PROVED (certified warm state) |
| $r_{\mathrm{extremal}}(5) = 0.7038$, $\mathrm{eff}(5) = 0.1739$ | PROVED (certified warm state) |
| Bottom-shell forced-negative structure at k=4 | CONFIRMED numerically |
| $C(I_k) \le T_{\mathrm{relay}}(k) \times (1-r_{\mathrm{ext}})/(1+r_{\mathrm{ext}})$ | **NOT PROVED** (contradicted at k=8) |
| k=8–12 closed via this formula | **FALSE** — see §3.2 |

---

## 5. The Actual Proof Gap and the Three Proof Routes

### 5.1 Gap summary

The relay tower gives $C(I_k) \le T_{\mathrm{relay}}(k) = 1.4601/2^{k/2}$.
The threshold is $C(I_2) = 0.022742$.

| $k$ | $T_{\mathrm{relay}}(k)$ | $T_{\mathrm{relay}}/C(I_2)$ | Current status |
|-----|------------------------|----------------------------|----------------|
| 8 | 0.09126 | 4.01× | Uncertified (Adam incumbent only) |
| 9 | 0.06453 | 2.84× | No certification started |
| 10 | 0.04563 | 2.01× | No certification started |
| 11 | 0.03226 | 1.42× | No certification started |
| 12 | 0.02281 | 1.003× | No certification started |
| ≥13 | — | <1× | PROVED by tail\_cert.py |

k=12 is the hardest: the relay tower misses the threshold by only 0.3%
($1.4601 > 1.4555 = C(I_2) \cdot 2^6$).

---

### 5.2 Joint Schur finite-block certificate

**Correct target:** The useful joint Schur statement is not the literal output-space
inequality

$$\langle F_{\mathrm{oa}}, F_{\mathrm{pmid}} \rangle_{D^{-1}} \le 0.$$

That output-space statement contains cross-fiber terms after convolution and is
stronger than what the tail proof needs.  The correct theorem is the
Schur/Hilbert--Schmidt version before the output-fiber expansion: after
symmetrising the two primitive input slots, the mixed Schur kernel for the two
Lions-paired tensors is non-positive.  Equivalently,

$$
\|F_{\mathrm{oa}}+F_{\mathrm{pmid}}\|_{\mathrm{Schur}}^2
\le
\|F_{\mathrm{oa}}\|_{\mathrm{Schur}}^2
+
\|F_{\mathrm{pmid}}\|_{\mathrm{Schur}}^2 .
$$

For fixed triad geometry write

$$
\xi+\eta+\zeta=0,
\qquad s=|\xi|^2,
\qquad t=|\eta|^2,
\qquad \rho=|\zeta|^2,
\qquad c=\xi\cdot\eta.
$$

Let $P_\zeta$ be orthogonal projection onto $\zeta^\perp$, and let
$a\in\xi^\perp$, $b\in\eta^\perp$.  Up to the scalar Lions coefficients, the
two paired tensors have the angular forms

$$
A_{\mathrm{oa}}(a,b)=(\eta\cdot a)P_\zeta b,
\qquad
A_{\mathrm{pmid}}(a,b)=(\zeta\cdot b)P_\zeta a.
$$

The Hilbert--Schmidt mixed trace over the two polarisation planes is

$$
T_{\mathrm{HS}}(s,t,\rho)
:=\sum_{a,b}\langle A_{\mathrm{oa}}(a,b),A_{\mathrm{pmid}}(a,b)\rangle
= q\,{(s+c)(t+c)\over \rho},
\qquad q:=1-{c^2\over st}.
$$

On the dyadic annulus $1\le s,t,\rho\le2$, this factor is non-negative because
$q\ge0$, $s+c=(\rho+s-t)/2\ge0$, and $t+c=(\rho+t-s)/2\ge0$.
The Lions scalar coefficients are $(s-\rho)$ and $(t-s)$.  After swapping the
primitive inputs $s\leftrightarrow t$, the mixed coefficient sum is

$$
(s-\rho)(t-s)+(t-\rho)(s-t)=-(s-t)^2\le0.
$$

Since $T_{\mathrm{HS}}$ is symmetric in $s,t$ and non-negative, the symmetrised
mixed Schur kernel is non-positive.  This proves the Schur-level joint
orthogonality and justifies combining the paired kernels quadratically.

The resulting scalar kernel is the pointwise joint kernel

$$K(s,t):=M_{\mathrm{oa}}(s,t)+M_{\mathrm{pmid}}(s,t),$$

not merely the sum of the two separate endpoint suprema.  Its endpoint value at
$(s,t)=(2,1)$ is

$$
K_* = K(2,1)
= {\sqrt2\,(110\log2+43)\over1280},
\qquad
\sqrt{K_*}=0.3629733751\ldots .
$$

The endpoint certificate is a direct Bernstein check.  Put $s=a^2$, $t=b^2$ and

$$
G(a,b):=1280a^7b^7\bigl(K_*-K(a^2,b^2)\bigr).
$$

After substituting $a=1+(\sqrt2-1)x$, $b=1+(\sqrt2-1)y$, the polynomial has
bidegree $(12,10)$.  Using $0.693<\log2<0.694$ coefficientwise, all Bernstein
coefficients of the resulting lower polynomial on $[0,1]^2$ are non-negative;
the only zero coefficient is the endpoint coefficient corresponding to
$(x,y)=(1,0)$, i.e. $(s,t)=(2,1)$.  Hence $K(s,t)\le K_*$ on $[1,2]^2$.
This audit is reproduced by `scripts/gap3/route_a_joint_kernel_cert.py`.

For k=12, the same script certifies the required finite-cell thickening.  With
$h=2^{-12/2}=1/64$ and
$\varepsilon=(\sqrt6+3h/4)h$, the square
$[1-\varepsilon,2+\varepsilon]^2$ is contained in
$a,b\in[49/50,10/7]$ after $s=a^2$, $t=b^2$.  Bernstein positivity on this
rational enlarged square proves

$$
|\partial_s K|\le {2\over5},
\qquad
|\partial_t K|\le {3\over5},
\qquad
K_{\mathrm{collar}}\le {3\over10}.
$$

Using $\log2\le347/500$ for the endpoint constant, the resulting finite relay
envelope satisfies

$$
U_{12}\le1.3034641156 < C(I_2)2^6=1.4554793862\ldots,
$$

leaving at least $0.1520152706$ of rescaled margin.

**Effect on the relay tower:** The ordered first-relay constant improves from

$$0.5732817578\ldots$$

to

$$C_{001,\mathrm{ord}}^*\le\sqrt{K_*}=0.3629733751\ldots.$$

Consequently the relay tower contribution improves to

$$0.3629733751\ldots\left(1+\sum_{m\ge1}2^{-3m/2}\right)
=0.5614901059\ldots.$$

Keeping the old primitive-core bound gives

$$0.5732817578\ldots+0.5614901059\ldots=1.1347718636\ldots,$$

which is below the k=12 threshold $C(I_2)2^6=1.4554793862\ldots$.
Applying the same joint paired-kernel bound to the primitive core gives the
asymptotic primitive-plus-relay constant

$$
0.3629733751\ldots\left(1+1+\sum_{m\ge1}2^{-3m/2}\right)
=0.9244634810\ldots,
$$

which is below the k=11 threshold $1.0291793439\ldots$ before finite-k
Riemann-sum error.

**Current joint Schur closure status:** The joint Schur finite-cell certificate
closes k=12 with comfortable finite-k margin and now closes k=11 by the weighted
row-sum shell/rho certificate in `references/K11_SHELL_RHO_LEMMA.md`.  It does
not by itself close k=8, k=9, or k=10: even the zero-error finite-height joint
tower is larger than those thresholds.  Those blocks are now closed by the
midpoint-centered row certificate in §5.3.  The same zero-error tower also misses
k=2--7; those blocks are closed only by their separate exact or finite numerical
certificates.

The k=11 obstruction is now reproducible in the audit script with
`--diagnose-k11`.  The k=11 thickening is covered by
$a,b\in[19/20,3/2]$, and Bernstein positivity on this square proves the simple
finite-cell bounds

$$
|\partial_sK|\le {1\over2},
\qquad
|\partial_tK|\le {3\over4},
\qquad
K_{\mathrm{collar}}\le {2\over5}.
$$

They give only

$$
U_{11}\le1.5104552225,
\qquad
C(I_2)2^{11/2}=1.0291793439\ldots,
$$

so the current finite-cell perturbation architecture cannot close k=11 by
constant polishing.  A separate shrink-factor diagnostic shows that the
k=11-scale derivative/collar perturbation would have to be reduced to roughly
20% of its present size to fit the threshold.  This points to a genuinely
sharper residue/Riemann-sum argument or a signature-specific primitive-core
bound, not a minor endpoint cleanup.

**New k=11 diagnostic target:** `scripts/gap3/k11_route_a_error_probe.py`
confirms that the obstruction is the vector-cube thickening scale, not the
joint endpoint kernel.  Sampling the exact k=11 thickened square gives roughly

```text
|partial_s K| <= 0.37112,  |partial_t K| <= 0.55754,  collar <= 0.27476,
```

but even these constants still give only `U_11≈1.37370`, above the threshold
`1.02918`.  If the same scalar kernel is charged on the actual squared-radius
shell/rho mesh instead — primitive spacing `h^2` and relay spacing
`(2^j h)^2` — then even the coarse certified constants
`(1/2,3/4,2/5)` give

```text
U_11,radial-coarse≈0.9595661322 < 1.0291793439,
```

leaving about `0.06961` of margin.  With the sampled exact-thickening constants
the diagnostic envelope improves to

```text
U_11,radial≈0.9499917244 < 1.0291793439,
```

leaving about `0.07919` of margin.  The needed replacement has now been carried
out at the row-sum level.  The all-row guard gives
`R_11<=1.011760776695`, below the squared radial budget `1.150356100192`.  The
integrated audit command

```powershell
python scripts/gap3/route_a_joint_kernel_cert.py --skip-bernstein --certify-k11-weighted-row
```

prints

```text
sqrt(R_11) * U_11,radial <= 0.965192260185
  < C(I_2)2^(11/2) = 1.029179343858,
final k=11 margin >= 0.063987083673.
```

Thus the remaining finite range after the joint Schur framework alone was k=8,
k=9, and k=10.  The midpoint-centered row certificate supplies the additional
finite estimate needed for those three blocks.

---

### 5.3 Midpoint-centered row certificate

**What it provides:** For k=8--10, replace the universal joint Schur output weight by
the midpoint-centered coefficient coming from $B(u,u,u)=0$, then combine the
centered scalar kernel with finite row guards and the existing low-block
certificates.  This supersedes the earlier signature-specific primitive-core
target for the purpose of closing the finite range.

**The P11 sign-forcing sub-route (even k):**

For even $k \ge 4$, the B-type bottom shell ($|n|^2 = 2^k$, 6 axis-aligned modes)
has forced negative contributions at the k=4 optimizer (r_shell up to 94 at those
shells). A naive extrapolation suggested the following lemma:

> **False as stated (sign-forcing, even k):** For any divergence-free
> field $u$ on $I_k$ (even $k \ge 4$), the aggregate contribution to $B(u,u,\Delta u)$
> from output modes $\ell$ with $|\ell|^2 \in \{2^k, 2^k+1, 2^k+2\}$ (the B-type
> bottom cluster) is **non-positive**.

This field-independent sign statement cannot be used as written.  The diagnostic
`scripts/gap3/bottom_cluster_sign_probe.py` evaluates exactly these bottom-output
triads.  After correcting the checkpoint replay to use the stored bounded
physical coordinates (`param_transform=direct`) and the full `±n` denominator,
the k=8 May 29 checkpoint gives a negative bottom cluster:

$$
\frac{B_{\mathrm{bottom}}}{X^2D}=-1.5935513975\times 10^{-3}<0,
\qquad
\frac{B_{\mathrm{bottom}}/(X^2D)}{R_{\mathrm{stored}}}=-7.72498\times 10^{-2}.
$$

A random 8-sample full-block sweep still produced both signs, so the blanket
field-independent lemma remains false.  Corrected wider low-band probes at the
checkpoint are suppressive: output shells `256..263` contribute
`-3.9205390015e-3` (`-19.0%` of the stored total), and `256..271` contributes
`-8.4248997741e-3` (`-40.8%` of the stored total).  The viable centered-row
replacement is therefore not a blanket bottom/low-band non-positivity lemma, but
a full-block signed-family envelope: identify which A/B/C/D family subwebs are
structurally suppressive and bound the positive high-output residue
quantitatively.

**Current stronger diagnostic (May 30): midpoint-centered output weights.** The
zeroth-order cancellation $B(u,u,u)=0$ appears in the trusted stream as

```text
sum_shell output_signed(shell)/shell = 0.
```

Thus the Laplacian weight may be recentered:

```text
B(u,u,Delta u) = sum_shell (shell - mu) * output_signed(shell)/shell.
```

For the k=8 checkpoint, taking `mu=384=3*2^(8-1)` makes every active A/B/C/D
family triple nonnegative after centering.  The centered family absolute sum
equals the signed total to floating precision:

```text
negative_centered_family_count = 0
min_centered_family = DDD 1.367917570626857e-07
centered_family_abs_sum = 2.062854038024919e-02
```

The same midpoint-centered family check gives no negative family triples for
k=6 and k=7 checkpoints; k=4 fails and k=5 fails only at a very small level.
The diagnostic has now been upgraded into a centered-kernel audit.  The global
identity, not the family-sign hypothesis, gives the raw output coefficient
`rho-3/2` after dyadic rescaling.  The scalar kernel

```text
K_c(s,t) = int_1^2 (rho-3/2)^2 * disc^2
           / (32 (st)^(7/2) rho^2) d rho,
disc = 4st - (rho-s-t)^2,
```

is certified by `scripts/gap3/route_b_centered_kernel_cert.py` on the enlarged
square `a,b in [1,10/7]`:

```text
K_c <= 1/60,
|partial_s K_c|, |partial_t K_c| <= 1/25,
collar <= 9/128.
```

The pure radial centered budget already fits k=8--10, with the delicate k=8
margin

```text
U_center,radial(k=8) <= 0.362954948758
  < C(I_2) 2^4 = 0.363869846549.
```

For a row-sum version with explicit slack, splice in the already-certified
reduced blocks instead of using the centered bound at the deepest levels.  With
coarse centered row guards `R<=5/4` on the even k8/k6 rows and `R<=3/2` for the
wide-margin odd/top rows, the mixed audit gives

```text
k= 8  U_mixed <= 0.340202543023 < 0.363869846549  margin +0.023667303527
k= 9  U_mixed <= 0.369886147671 < 0.514589671929  margin +0.144703524258
k=10  U_mixed <= 0.374439743029 < 0.727739693099  margin +0.353299950070
```

The earlier family-sign route has been replaced by explicit row guards.  The
high-precision row verifier `scripts/gap3/centered_row_mp_verify.py` counts exact
`(m,ell)` cells and, in `--interval` mode, reports
`actual_upper/continuum_lower`.  It checks the needed even rows against `R<=5/4`:

```text
k8 all rows: worst n=438, R=1.028818401493566
k6 all rows: worst n=88,  R=1.103121376559912
k4 all rows: worst n=24,  R=1.228942470683844
```

The small odd reduced block k7 is also interval-checked against `R<=3/2`, while
the larger wide-margin top blocks have complete fast all-row screens far below
the same guard:

```text
k7 all rows: worst n=163, R=1.037908796942949  (interval)
k9 all rows: worst n=512,  R=1.007609447255  (guarded coarse)
k10 all rows: worst n=1968, R=1.004985996223  (guarded coarse)
```

The row evidence is therefore far below the mixed-budget guards.  The tight k8
guard now has interval backing, and the wide-margin k9/k10 top guards have a
guarded coarse certificate with explicit `eps_g=1e-8`, `eps_k=1e-10` allowances
from `scripts/gap3/centered_row_coarse_cert.py`.  The family one-sidedness
evidence remains useful intuition, but it is no longer the shortest closure
route.  A blanket field-independent family version is false: a random full k6
field with seed `12345` has `32` negative centered family triples at `mu=96`.

Connection: `thm:crystal_evenk` in `paper2/ns_cancellation.tex` remains useful
structural background for the older bottom-shell route, but it is not needed for
the centered row certificate.  The odd-k issue is likewise bypassed: the
midpoint-centered kernel and row guards treat k9 directly, without a separate
bottom-shell sign-forcing lemma.

**Status:** The midpoint-centered row certificate closes k=8--10 at the working
certificate level.  The joint Schur and weighted-row finite certificates remove
the k=11 and k=12 bottlenecks, and the large-k tail covers k>=13.

---

### 5.4 Route C — Direct GPU global certification, block by block

**What it requires:** For each k=8, 9, ..., 12 in turn:
1. Run Adam optimizer to find a high-R warm start.
2. Verify the warm-start value via mpmath (50+ decimal places).
3. Compute and certify the Hessian (negative definite → local max confirmed).
4. Run a global scan (many random starts) to check for better maxima.
5. If the globally certified $C(I_k) < C(I_2)$, that block is closed.

**Computational cost:** The dominant cost is the triad evaluation, which scales as
$O(N_k^2)$ where $N_k \sim 2^{3k/2}$ is the number of modes in $I_k$.

| $k$ | Modes ($N_k$) | Triads (approx) | Estimated cert time |
|-----|--------------|-----------------|---------------------|
| 8 | 15,709 | 166,713,936 | ~3 days (GPU) |
| 9 | ~56,000 | ~1.3 billion | ~weeks |
| 10 | ~195,000 | ~10 billion | months |
| 11 | ~680,000 | ~80 billion | impractical |
| 12 | ~2.4 million | ~650 billion | impractical |

The exponential growth ($\times 8$ in triads per step in k) makes k≥10 effectively
intractable by direct numerical certification at current GPU speeds. Route C is
therefore only viable for k=8, and possibly k=9 with a very efficient warm start.

**k=8 status:** The Adam run has a float-precision incumbent (~0.020629) that can
serve as a warm start for mpmath verification. The full global cert is ~3 days.
**Do NOT restart the Adam run** — use the existing checkpoint as the warm-start origin.

**k=9–10:** Direct optimizer certification is not feasible in reasonable time.
They are instead closed by the midpoint-centered row certificate.

---

### 5.5 Recommended priority

1. **Make the joint Schur certificate paper-ready.** The correct statement is the Schur-level joint
  paired-kernel theorem in §5.2, not the literal output-space cross-term claim.
  The audit script now verifies the Hilbert--Schmidt trace identity, the
  Bernstein endpoint certificate for $M_{\mathrm{oa}}+M_{\mathrm{pmid}}$, and
  the k=12 finite-cell constants, and the weighted-row k=11 certificate.  The
  remaining formal task is to translate those certificates into paper prose.

2. **Make the midpoint-centered certificate paper-ready.** The centered kernel certificate and row guards
  now close k=8--10.  The remaining work is exposition and final audit packaging,
  not a new optimizer campaign.

3. **Keep Route C only as optional optimizer evidence.** The k=8 Adam checkpoint
  is useful structural data, but it is no longer needed to close the theorem.

  Use `references/SMALL_K_STRUCTURAL_ATLAS.md` as the small-block mechanism
  guide.  The purpose of k2--k7 at this stage is not to analytically replace
  every finite certificate; it is to support the centered finite proof.  k2 and
  k3 are the exact/reduced models, while k4--k7 are certified low-block inputs
  or row-guard sanity checks in the mixed budget.

---

## 6. Proof Status Summary

### What is proved

| Block | Status | Method |
|-------|--------|--------|
| k=1 | $C(I_1) = 0$ | Analytic (exact) |
| k=2 | $C(I_2) = 0.022741865409341\ldots$ | Exact closed form |
| k=3 | $C(I_3)\le0.021936470<C(I_2)$ | Refined algebraic/nucleus comparison certificate; $K_3$ candidate is $0.021936469459403\ldots$ |
| k=4 | $C(I_4) = 0.021064396605547\ldots$ | GPU cert, 50 dps |
| k=5 | $C(I_5) = 0.018479317637642\ldots$ | GPU cert, 50 dps |
| k=6 | $C(I_6) = 0.020443793141444\ldots$ | mpmath + Hessian |
| k=7 | $C(I_7) = 0.020280600900469\ldots$ | GPU cert May 2026 |
| k=8 | $C(I_8) < C(I_2)$ | Midpoint-centered mixed certificate: $U_{mixed}\le0.340202543023$ |
| k=9 | $C(I_9) < C(I_2)$ | Midpoint-centered mixed certificate: $U_{mixed}\le0.369886147671$ |
| k=10 | $C(I_{10}) < C(I_2)$ | Midpoint-centered mixed certificate: $U_{mixed}\le0.374439743029$ |
| k=11 | $C(I_{11}) < C(I_2)$ | Weighted-row joint Schur certificate: $\sqrt{R_{11}}U_{11,radial}\le0.965192260185$ |
| k=12 | $C(I_{12}) < C(I_2)$ | Joint Schur finite-cell certificate: $U_{12}\le1.303464115585$ |
| k≥13 | $C(I_k) < C(I_2)$ | tail\_cert.py: $U(13) < 1.9106 < 2.0584$ |

### What is not used as proof input

| Block | Data not used | Reason |
|-------|---------------|--------|
| k=8 | Adam incumbent `0.020628554...` | float checkpoint only; closure uses the midpoint-centered upper bound |
| k=9 | old value `0.006520` | stale restricted scan; closure uses the midpoint-centered upper bound |
| k=10 | old value `0.004165` | stale restricted scan; closure uses the midpoint-centered upper bound |

**No finite holes remain in the working proof chain:** k=1--7 are exact or
finite-certified, k=8--10 are closed by the midpoint-centered certificate,
k=11--12 by the joint Schur/weighted-row certificates, and k>=13 by the tail
certificate.

### Immediate next steps (in priority order)

1. **Finish the joint Schur certificate in paper-ready form.** Use the Schur/Hilbert--Schmidt
  symmetrised mixed-kernel proof and the joint endpoint certificate from §5.2.
  This closes k=11 and k=12 and gives the correct constants for any later
  finite-tail sharpening.

2. **Finish the midpoint-centered certificate in paper-ready form.** State the midpoint shift from
  $B(u,u,u)=0$, cite the centered kernel audit, and include the row-guard
  certificates for k=8--10.

---

## 7. Appendix: r_needed Calculation

For the P11 formula to give a valid bound $C(I_k) \le T_{\mathrm{relay}}(k) \times \mathrm{eff}
< C(I_2)$, we would need both:
(a) $T_{\mathrm{relay}}(k) \ge T_{\mathrm{field}}(u_{\mathrm{opt}})$ (so the formula is valid), AND
(b) $r_{\mathrm{extremal}}(k) \ge r_{\mathrm{needed}}(k)$ where:

$$r_{\mathrm{needed}}(k) = \frac{1 - \alpha(k)}{1 + \alpha(k)},
\quad \alpha(k) = \frac{C(I_2)}{T_{\mathrm{relay}}(k)}.$$

| $k$ | $T_{\mathrm{relay}}(k)$ | $\alpha(k)$ | $r_{\mathrm{needed}}(k)$ | Condition (a) holds at k=4? |
|-----|------------------------|-------------|--------------------------|----------------------------|
| 8 | 0.09125636 | 0.24921 | 0.60101 | NO (T_relay < T_field for k≥6) |
| 9 | 0.06452799 | 0.35243 | 0.47882 | NO |
| 10 | 0.04562818 | 0.49842 | 0.33474 | NO |
| 11 | 0.03226399 | 0.70508 | 0.17311 | NO |
| 12 | 0.02281409 | 0.99706 | 0.00148 | NO |

Condition (a) fails for all k=8–12 at the scale where the optimizer lives.
These $r_{\mathrm{needed}}$ values are therefore useful only as hypotheses for
a **different bound** on $T_{\mathrm{field}}$ (not the relay tower).

This historical calculation is no longer the closure route.  The needed new
tool is the midpoint-centered row certificate of §5.3, not a P11 efficiency
multiplier.

---

## 8. How This Relates to SPECTRAL_TAXONOMY_TAIL_REPAIR.md

| Topic | SPECTRAL_TAXONOMY_TAIL_REPAIR.md | This document |
|-------|----------------------------------|---------------|
| Relay tower derivation (Lemmas 1–5, Lions pairing) | Full proof | Summarised in §1.3 |
| Finite-k certificate (tail_cert.py) | Full cert (k≥13) | Status table in §6 |
| Primitive core bound $C_{000,\mathrm{pair}}^*$ | Derived and proved | Referenced in §5.1 |
| Schur-level joint paired kernel | Setup in Lions-paired recombination | Corrected and quantified in §5.2 |
| Midpoint-centered row certificate | Supported by centered output identity and taxonomy | §5.3 closes k=8--10 |
| P11 inter-triad cancellation ratio $r$ | Not treated | **Full treatment here** |
| Bottom-shell sign-forcing mechanism | Not treated | §4.1 and §5.2 |
| k=8 formula contradiction | Not in document | §3.2 explicitly |

The two documents are COMPLEMENTARY. SPECTRAL_TAXONOMY gives the relay-tower and
centered-kernel machinery.  This document analyses the additional P11
cancellation structure, records why the naive P11 efficiency formula fails, and
now records the replacement midpoint-centered certificate that closes k=8--10.

Do not use the naive relay-tower/P11 efficiency formula as a proof of k=8--12.
Use the midpoint-centered row certificate for k=8--10, the joint Schur/weighted-
row certificates for k=11--12, and the tail certificate for k>=13.

---

*End of document. Last updated: 2026.*
