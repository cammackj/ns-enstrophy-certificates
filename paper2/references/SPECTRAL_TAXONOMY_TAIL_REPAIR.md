# Spectral Taxonomy Tail Repair

Working memo, not paper text.

**Companion document:** `references/P11_CANCELLATION_BOUND.md`
Covers the Principle-11 inter-triad cancellation analysis, terminology disambiguation,
and the corrected proof status for k=8-12. Read that document before attempting
to apply the relay tower to individual block cancellation arguments.

**Paper theorem ledger:** `references/PAPER_THEOREM_LEDGER.md`
This is the paper-facing dependency map: theorem names, proof-chain assembly,
certified constants, audit commands, and manuscript-integration gate.

## Goal

Replace the superseded single-plateau assertion

```tex
C_B^* <= 0.157
```

with a weaker but structurally honest large-k tail theorem. The theorem only
needs to keep every dyadic block below the exact k=2 value

```tex
C(I_2) = 0.022741865409341...
```

It does not need to recover the old sharp-looking plateau constant.

## Current Proof Package Status

As of May 30, 2026 the working proof chain has no finite dyadic holes:

| Range | Closure mechanism | Status |
|-------|-------------------|--------|
| k=1--7 | exact, mpmath, or GPU finite certificates | closed |
| k=8--10 | midpoint-centered row certificate | closed as upper bounds below `C(I_2)` |
| k=11--12 | joint Schur and weighted-row finite certificates | closed |
| k>=13 | effective tail certificate | closed |

This memo is now the paper-ready staging document for the replacement theorem
package.  The exact optimizer values for k=8--10 are not claimed; those blocks
are closed by certified upper bounds.  The companion note
`references/P11_CANCELLATION_BOUND.md` records the diagnostic history and the
same finite-block status in shorter form.

## Current Hypothesis

The sequence `C(I_k)` is not governed by one monotone law in k. It is governed
by monotone envelopes attached to spectral-shell mechanism families. The visible
spikes and drops occur when the optimizer changes family signature:

- B/A low block behavior at k=2.
- D/C/A nucleus behavior at k=3.
- Augmented resonant webs at k=4 and k=5.
- Diffuse mixed A/B/C/D webs at k=6 and k=7.
- A/B exceptional mechanisms such as the old k=9 candidate.
- Possible overlap exceptions when a shell has multiple raw arithmetic features,
  even if the paper's priority taxonomy assigns it one label.

Thus the replacement statement should not be "C_res(k) is monotone." It should
be "each primitive feature-signature envelope has a monotone or bounded tail,
and the global value is the maximum over finitely many such envelopes."

## Important Distinction: Labels vs Raw Features

The paper's shell taxonomy is priority ordered:

1. D-type: `n = 2a^2`.
2. A-type: not D, even, and admits an equilateral shell triad.
3. C-type: not D/A, and `n = a^2 + b^2` with `a > b >= 1`.
4. B-type: none of the above.

For proof purposes, we should track the raw feature vector, not only the final
priority label:

```text
Q_D(n): n = 2a^2
Q_A(n): shell n admits an equilateral triad
Q_C(n): n = a^2 + b^2, a > b >= 1
```

The priority label is useful for exposition, but overlap shells are exactly where
exceptions may live. D-shells, for example, have special equilateral/crystal
geometry and should not be flattened into generic A-type behavior.

## Primitive Feature Signatures

For the tail theorem, a signature should be attached to the nonvanishing triad
web, not merely to a list of occupied shells. Let

```tex
Q(n):=(Q_D(n),Q_A(n),Q_C(n))\in\{0,1\}^3.
```

For an ordered primitive same-valuation triad `(ell,r,s)` in the GPU/paper
ordering, with `ell=r+s` in the implemented tensor and
`nu_2(ell)=nu_2(r)=nu_2(s)=0`, define its raw ordered type by

```tex
T(ell,r,s):=\bigl(Q(|ell|^2),Q(|r|^2),Q(|s|^2),\pi(|ell|^2,|r|^2,|s|^2)\bigr),
```

where `pi` records the shell-order pattern and the slot on which the Laplacian
weight lands. Triads killed by the same-shell vanishing
`|ell|^2=|r|^2=|s|^2` are omitted from `T`.

Important correction: the previously tempting ULU build-time deletion
`|ell|^2=|r|^2` is invalid for the ordered numerator.  A k6 replay shows that
the ULU aggregate can be nonzero.  Thus ULU triads must remain in the primitive
feature signature unless a separate field-specific or paired-sum identity is
proved.

For a valuation-0 field `u`, define its primitive feature signature by

```text
Sigma(u) = { T(ell,r,s) : u_hat[ell] u_hat[r] u_hat[s] != 0
             and the ordered triad is not one of the vanished types }.
```

This definition has two useful properties.

1. It is finite at the proof level: each shell contributes one of eight raw
   feature vectors, and the remaining order/slot data are finite.
2. It is stable under exact dyadic descendants, because `Q_D`, `Q_A`, and `Q_C`
   are stable under `n -> 4n`, while the shell-order pattern and tensor slot do
   not change.

Thus a family envelope can be indexed by a finite set of ordered primitive triad
types `sigma`, with `\Sigma(u)\subset\sigma`. The exact descendant lemma then
prevents a fixed primitive signature from generating larger rescaled values by
simple doubling; any new spike must come from a genuinely new primitive ordered
triad type or an overlap of such types.

## Verified Structural Clues

Direct shell-energy inspection of the certified xstar files gives:

### k=6

```text
C(I_6) = 0.02044379314144455
C_res(6) = 0.1635503451315564
```

The active support is a diffuse mixed web, not a bottom B-shell ladder. The
largest shell-energy fractions are:

```text
65(C) 10.52%, 69(B) 8.01%, 66(B) 6.18%, 110(B) 4.32%,
121(B) 4.24%, 125(C) 4.15%, 77(B) 3.33%, 83(B) 3.15%,
89(C) 3.15%, 74(A) 3.15%.
```

At the 1% shell-energy threshold, k=6 uses B, C, A, and D shells. This rules
out a one-ladder even-k model.

### k=7

```text
C(I_7) = 0.020280600900469462
C_res(7) = 0.22944880677215934
```

The active support is again a diffuse mixed web. The largest shell-energy
fractions are:

```text
129(B) 3.84%, 134(A) 3.43%, 146(A) 3.17%, 131(B) 3.04%,
145(C) 3.00%, 161(B) 2.81%, 149(C) 2.20%, 133(B) 2.07%,
153(C) 2.06%, 150(A) 2.05%.
```

The D bottom-shell fact is real but not the whole mechanism; D contributes to
the web rather than dominating the shell energy.

### k=8 guardrail checkpoint

The current dense k=8 Adam checkpoint gives:

```text
C(I_8) >= 0.0206285547465086
C_res(8) >= 0.3300568759441376
```

This is still below the exact raw k=2 value, but it is another diffuse mixed
web. Its largest shell-energy fractions are:

```text
261(C) 2.32%, 269(C) 2.15%, 257(C) 2.12%, 266(A) 2.09%,
281(C) 1.88%, 270(B) 1.87%, 290(C) 1.55%, 285(B) 1.52%,
293(C) 1.37%, 314(A) 1.34%.
```

At the 0.5% shell-energy threshold, the family mass is approximately:

```text
A: 14.05%, B: 25.40%, C: 22.26%, D: 0.51%.
```

So k=8 is not a single-family anomaly either. The finite exceptional blocks seem
to be broad primitive relay webs whose internal shell labels mix A/B/C/D.

### 2-adic layer profile

The same xstar/checkpoint states are concentrated in the primitive valuation
layer. Shell-energy fractions by vector valuation `nu_2(p)` are:

```text
k=6: v2=0 85.39%, v2=1 11.94%, v2=2 0.41%, v2=3 2.26%.
k=7: v2=0 87.04%, v2=1 11.00%, v2=2 1.21%, v2=3 0.75%.
k=8: v2=0 86.97%, v2=1 11.35%, v2=2 1.48%, v2=3 0.13%, v2=4 0.07%.
```

Thus the dangerous mechanisms are not high 2-adic descendants. They are
primitive-minimum webs with small relay layers, especially `(0,0,0)` and
`(0,0,1)` valuation patterns.

The current k=8 streaming replay rebuilds the same-shell-only filter on the fly,
does not use the stale version-0 triad cache, and matches the stored checkpoint:
`166,705,632` triads, signed ratio `2.062854038024918e-02`, stored ratio
`2.062855474650860e-02`, relative replay `0.9999993036`.  Grouping the same
stream by sorted valuation triples gives:

```text
(0,0,0): 0.01331502545538219  (64.55% of signed total)
(0,0,1): 0.00605865809908123  (29.37%)
(0,0,2): 0.00079114593446199  ( 3.83%)
(1,1,1): 0.00023417961232177  ( 1.14%)
(1,1,2): 0.00010698398386008  ( 0.52%)
(0,0,3): 0.00006846490812847  ( 0.33%)
(0,0,4): 0.00003559402140937  ( 0.17%)
```

No forbidden pattern such as `(0,1,1)` appears in the current stream, consistent
with Lemma 3.  The dominant non-primitive relay to control remains `(0,0,1)`,
followed by a tail estimate for higher gaps.  The replay arrays are saved in
`scripts\gap3\results\2026-05-29\k8_full_family_stream_direct.npz`.

These current-filter valuation numbers are now valid diagnostic data from the
trusted streaming replay, not cache-derived percentages.  Thus the first
genuinely new finite-block bound to prove is still the primitive relay norm for
`(0,0,1)`, followed by a tail estimate for `j>=2`.

## Deferred GPU Scan Modification Plan

The valuation/taxonomy structure can also guide future GPU scans for every k,
not just k=8. This should be an engineering optimization layer, not a change to
the mathematical source of truth.

The safe architecture is:

1. Use A/B/C/D raw features and 2-adic valuation layers to build a predicted
   support graph before full optimization.
2. Start exploration on the primitive core and the first relay layer:

   ```text
   (0,0,0) plus (0,0,1)
   ```

   with candidate shells chosen by primitive feature signatures and relay
   closure, not by blind full-block random starts.
3. Expand by relay closure only when a shell has negative ISC/gradient or strong
   coupling to the current optimizer.
4. Promote only promising starts/supports to the full complete-triad objective.
5. Keep the dense full objective as the verifier for final values, checkpoints,
   and certificates.

This is broader than simply filtering k=8 triads. The goal is to triangulate the
right basin for each k using the same arithmetic that appears in the proof:
primitive feature signatures, valuation layers, and relay closure. The GPU then
spends most of its time on plausible basins instead of uniformly sampling the
entire block.

Implementation should wait until the analytical structure is stable. When ready,
the first safe prototype is a read-only cache/support tool that writes proposed
valuation/taxonomy support sets and benchmarks whether they reproduce known
k6/k7/k8 basins under the full objective.

## Structural Guardrails

The following are methodological guardrails, not theorem statements.

- Numerical optima are structural hypotheses. The k6/k7/k8 diffuse webs should
   constrain the proof, not replace it.
- Every upper bound needs a tightness or falsification check. If a Schur or
   Hilbert--Schmidt estimate overshoots the observed relay contribution
   by orders of magnitude, it is probably not the structural bound we want.
- Principle 10: no term omission. A replacement for `lem:plateau` must account
   for primitive core, first relay, higher relays, and same-valuation descendants.
- Principle 11/13: look for actual cancellation or involutions before taking
   absolute values over all triads. Triangle inequalities are allowed as safe
   majorants, but they should be treated as deliberately lossy.
- Principle 12: classify by structural character. The A/B/C/D raw features and
   2-adic valuation layers are the right variables; a single monotone curve in
   `k` is the wrong model.

## Core Lemmas To Prove

### Lemma 1: Type stability under four-scaling

For every integer shell n, the raw feature vector is stable under `n -> 4n`:

```text
Q_D(n) iff Q_D(4n)
Q_A(n) iff Q_A(4n)
Q_C(n) iff Q_C(4n)
```

Proof.

For `Q_D`, the equivalence is immediate in one direction:

```tex
n = 2a^2 \quad\Longrightarrow\quad 4n = 2(2a)^2.
```

Conversely, if `4n = 2b^2`, then `b^2 = 2n` is even, so `b=2a`. Hence
`4n = 8a^2`, and therefore `n=2a^2`.

For `Q_C`, scaling a representation gives

```tex
n=a^2+b^2 \quad\Longrightarrow\quad 4n=(2a)^2+(2b)^2.
```

Conversely, suppose `4n=a^2+b^2` with `a>b>=1`. Squares are `0` or `1` modulo 4.
Since `a^2+b^2` is `0` modulo 4, both `a` and `b` must be even. Write
`a=2a_0`, `b=2b_0`; then `n=a_0^2+b_0^2`, with `a_0>b_0>=1`.

For `Q_A`, use the invariant formulation: shell `n` is A-raw if there are
integer vectors `p,q,r` with

```tex
p+q+r=0, \qquad |p|^2=|q|^2=|r|^2=n.
```

The forward direction is again scaling. Conversely, suppose such a triad exists
on shell `4n`. For each vector, `|p|^2` is divisible by 4. In three dimensions,
the number of odd coordinates of `p` is congruent to `|p|^2` modulo 4; since
this number is one of `0,1,2,3`, it must be `0`. Thus every coordinate of `p`,
`q`, and `r` is even. Dividing the triad by 2 gives an equilateral shell triad
on `n`.

Therefore every shell belongs to a dyadic primitive chain

```text
n0, 4 n0, 16 n0, ...
```

with fixed raw feature signature.

### Lemma 2: Exact value scaling on doubled supports

If a field v on `I_k` is doubled to a field `v^(2)` on `I_{k+2}` by

```text
p -> 2p,
```

then

```tex
R(v^{(2)}) = \frac12 R(v).
```

Proof.

Let `S_2v=v^(2)` be the Fourier field obtained from `v` by moving each
coefficient at wavevector `p` to wavevector `2p`, without changing its complex
vector coefficient. The divergence-free condition is preserved because
`u_p perp p` implies `u_p perp 2p`, and resonant triads are preserved because
`p+q+r=0` if and only if `2p+2q+2r=0`.

The numerator `B(v,v,Delta v)` contains one first derivative and one Laplacian,
so the dilation `p -> 2p` gives

```tex
B(S_2v,S_2v,\Delta S_2v)=2^3 B(v,v,\Delta v).
```

The denominator factors scale as

```tex
X(S_2v)^2 = 2^2 X(v)^2, \qquad D(S_2v)=2^2D(v),
```

because `X^2=||nabla v||_2^2` has two derivatives in total, while
`D=||Delta v||_2` has two derivatives before taking the norm. Hence

```tex
R(S_2v)
= \frac{2^3 B(v,v,\Delta v)}{(2^2 X(v)^2)(2^2D(v))}
= 2^{-1}R(v).
```

Thus descendants of an already observed mechanism cannot create a raw-value
spike; they decay by a factor `1/2` every two k-steps.

Equivalently, exact descendants preserve the rescaled value:

```tex
C_{\mathrm{res}}(k+2;S_2v)
= R(S_2v)2^{(k+2)/2}
= R(v)2^{k/2}
= C_{\mathrm{res}}(k;v).
```

So the useful monotonicity is raw decay of `C(I_k)`, not decay of every rescaled
descendant constant.

### Lemma 3: Primitive support decomposition

Every mode has a 2-adic primitive representative. Decompose a field by the
minimum 2-adic valuation of its active modes. Triad resonance imposes a strong
valuation rule: in any resonant triple, the minimum valuation must occur at
least twice.

Proof of the valuation rule.

For a nonzero integer vector `p`, define

```tex
\nu_2(p):=\min_i \nu_2(p_i),
```

with the convention `nu_2(0)=infty` for zero coordinates. Let `p+q+r=0` be a
resonant triad of nonzero integer vectors. If the minimum of
`nu_2(p),nu_2(q),nu_2(r)` occurred uniquely, divide the triad equation by the
corresponding power of 2. Modulo 2, the unique minimum vector has at least one
odd coordinate, while the other two vectors are coordinatewise even. Their sum
cannot be zero modulo 2. Contradiction.

Therefore the minimum 2-adic vector valuation in any resonant triad occurs at
least twice.

Two immediate consequences matter for the tail proof:

1. A triad containing exactly one primitive/minimum-scale mode is impossible.
2. Cross-scale triads have a hierarchical form: after factoring out the common
   minimum scale, at least two entries are primitive. This gives a finite
   primitive interaction graph modulo dyadic dilation.

This is the arithmetic replacement for the old continuum non-degeneracy
assumption.

### Lemma 4: Valuation-stratified relay reduction

Let

```tex
v = \sum_{a\geq0} v_a,
```

where `v_a` is supported on modes with vector valuation `nu_2(p)=a`. Then every
nonzero triad contribution belongs to one of the forms

```text
(a,a,a),       all three modes at the same valuation layer;
(a,a,a+j),     two minimum-scale modes and one higher-valuation relay, j>=1,
```

up to permutation of the three trilinear slots.

Proof. This is exactly Lemma 3 applied to each resonant triad: the minimum
valuation cannot occur once, so it occurs either twice or three times.

The all-equal part is an exact dyadic descendant. Factoring `2^a` out of every
wavevector maps its contribution to the corresponding primitive block and gives

```tex
R(v_a) = 2^{-a} R(v_a^{\mathrm{prim}}).
```

Therefore all layers with `a>=1` are automatically smaller than their primitive
ancestors. The only part capable of producing a new tail constant is the relay
family `(a,a,a+j)`. Factoring out the common minimum scale gives another factor
`2^{-a}`, so it is enough to bound the primitive-minimum relay operators

```text
(0,0,j),  j >= 1.
```

This is the precise reduction behind the taxonomy strategy: the large-k problem
does not require controlling arbitrary cross-scale chaos. It requires controlling
primitive A/B/C/D webs plus their finite or summable higher-valuation relays.

### Lemma 5: 2-adic relay congruence

In a primitive-minimum relay triad of type `(0,0,j)`, write

```tex
p+q+2^j h=0,
```

where `nu_2(p)=nu_2(q)=0` and `nu_2(2^j h)=j`. Then

```tex
p+q \equiv 0 \pmod{2^j}.
```

Equivalently,

```tex
q \equiv -p \pmod{2^j}.
```

Consequences:

1. For `j=1`, the two primitive input modes have the same parity vector modulo
   2, since `-p=p` over `F_2`.
2. For `j>=2`, the input pair is confined to a single opposite residue class
   modulo `2^j`. This is a strong thinning inside the shell-pair incidence
   count.
3. The relay primitive `h` lies in the lower dyadic block `I_{k-2j}` whenever
   `2^j h` lies in `I_k`. Thus the relay feature signature is inherited from a
   lower primitive block and is stable under Lemma 1.

This congruence is one analytic lever, but it should not be overread by itself:
once the relay mode `2^j h` is fixed, the relation `q=-p-2^j h` makes the
congruence automatic. The real tail mechanism is the combination of:

1. the lower-density relay set `2^j h` with `h` in `I_{k-2j}`;
2. the parity/congruence restriction on the two primitive inputs;
3. the existing rank-2 shell/hyperplane Gram count.

The remaining analytic task is therefore to turn this into valuation-restricted
continuum constants for the ordered relay operators. The `(0,0,1)` ordered
constant is now bounded above; `j>=2` should inherit it through a summable
density tail.

### Theorem Target: valuation-restricted relay constants

For each `j>=1`, define the rescaled primitive relay constant

```tex
C_{00j,ord}^{(k)}
:= 2^{k/2}
\sup_{u,w\ne0}
\frac{|B_{0,0,j}(u,u,w)|}{X(u)^2D(w)},
```

where `u` is supported on valuation-0 modes in `I_k`, `w` is supported on
valuation-`j` modes in `I_k`, and `B_{0,0,j}` includes all ordered trilinear
placements of the valuation pattern. The theorem we need is a decomposition

```tex
\limsup_{k\to\infty} C_{001,ord}^{(k)} \leq C_{001,ord}^*,
```

with an explicit usable bound for `C_{001,ord}^*`, plus a summable tail estimate

```tex
\sum_{j\geq2} \sup_k C_{00j,ord}^{(k)} \leq C_{00,\ge2,ord}^*.
```

The k=8 decomposition says this is the right split: `(0,0,1)` carries about
`0.00606` of the raw ratio, while all `j>=2` primitive relays together carry
less than `0.001` at the current checkpoint.

Rescaled to the `C_res` scale, the current k=8 checkpoint contributions are
approximately:

```text
(0,0,1): 0.00605866 * 16 = 0.09694
j>=2:    0.00089520 * 16 = 0.01432
```

### Theorem Target: valuation-density scaling

Let `rho_j` be the asymptotic density, relative to the full integer lattice, of
modes with vector valuation `j`:

```tex
\rho_j = \frac{7}{8}\,2^{-3j}.
```

The valuation-0 input lattice has density `rho_0=7/8`. The valuation-`j` relay
lattice has density `rho_j`. In the continuum normalization of
`B/(X^2D)`, the valuation-0 input density cancels against `X(u)^2`, while the
relay density survives through `D(w)` as a square-root factor. Thus the useful
upper-bound target is

```tex
C_{00j,ord}^* \leq 2^{-3(j-1)/2} C_{001,ord}^*, \qquad j\geq1.
```

Equality may hold in the ideal equidistributed continuum model, but the proof
only needs the upper bound. The theorem to prove is: after rescaling to the
annulus `S={1<=|xi|^2<2}`, the geometry of the ordered `(0,0,j)` relay is
controlled by the same Lions-paired continuum kernels as `(0,0,1)`, and the only
extra loss is the square-root relay-density factor.

Proof skeleton.

1. Residue count. Fix a primitive input mode `p` and `j>=1`. The condition
   `nu_2(p+q)=j` is equivalent modulo `2^{j+1}` to

   ```tex
   q = -p + 2^j h, \qquad h\in(\mathbb{Z}/2\mathbb{Z})^3\setminus\{0\}.
   ```

   Thus there are `7` admissible residue classes modulo `2^{j+1}`, i.e. density

   ```tex
   7/2^{3(j+1)} = (7/8)2^{-3j} = \rho_j
   ```

   for the second primitive input once the first is fixed.
2. Continuum equidistribution. On the rescaled annulus, those residue classes
   are equidistributed as `k -> infinity`, so the ordered `(0,0,j)` lattice sum
   is bounded by the same Lions-paired continuum kernels as `(0,0,1)`,
   multiplied by the density ratio `rho_j/rho_1` at the numerator level.
3. Normalization. The relay norm satisfies

   ```tex
   D_j(w)^2 \sim \rho_j\int_S |\xi|^4 |w(\xi)|^2d\xi,
   ```

   so replacing `j=1` by general `j` contributes the square-root factor

   ```tex
   (\rho_j/\rho_1)^{1/2}=2^{-3(j-1)/2}.
   ```

This gives the desired upper bound for `C_{00j,ord}^*` in terms of
`C_{001,ord}^*`.

### Working proof: residue Riemann sums for valuation-density scaling

The density-scaling theorem should be proved as a nonnegative-kernel Riemann-sum
comparison, not as a heuristic count. After rescaling `p=2^{k/2}\xi`, the
dyadic block becomes the fixed annulus `S={1<=|xi|^2<2}` with mesh
`h=2^{-k/2}`.

Residue Riemann-sum lemma. Let `Omega` be a bounded Jordan-measurable subset of
`R^d`, let `A` be a fixed subset of `(Z/MZ)^d`, and let `Phi` be continuous and
nonnegative on a neighborhood of `Omega`. Then

```tex
h^d\sum_{n\in h^{-1}\Omega,\ n\bmod M\in A}\Phi(hn)
\longrightarrow \frac{|A|}{M^d}\int_\Omega\Phi(x)\,dx.
```

Proof. Decompose the sum into residue classes `a in A`. For fixed `a`, write
`n=a+Mm`. Then

```tex
h^d\sum_m \Phi(h(a+Mm))
=\frac{1}{M^d}(Mh)^d\sum_m \Phi(ha+Mh\,m),
```

which is the ordinary Riemann sum on the translated mesh `Mh Z^d`, multiplied
by `M^{-d}`. The translation `ha` tends to zero, so the limit is
`M^{-d} int_Omega Phi`. Summing over `A` proves the claim. Boundary errors are
controlled by the Jordan-measurability of `Omega`; for upper bounds one may also
thicken/shrink the annulus by `epsilon` and let `epsilon -> 0`.

Apply this lemma in dimension six to the pair lattice `(p,q)`. The ordered relay
kernels are continuous on the compact annulus constraint because
`rho=|xi+eta|^2` is restricted to `[1,2]`, so the factors `rho^{-2}` and
`rho^{-3}` are harmless.

For fixed primitive `p` and `j>=1`, the condition `nu_2(p+q)=j` is equivalent to

```tex
q=-p+2^j h \pmod {2^{j+1}},
\qquad h\in(\mathbb Z/2\mathbb Z)^3\setminus\{0\}.
```

Thus the admissible `q` residues have density

```tex
\rho_j=7/2^{3(j+1)}=(7/8)2^{-3j}.
```

This is the whole valuation loss in the pair incidence. Since the first input is
already restricted to the primitive lattice of density `rho_0=7/8`, the input
normalization by `X(u)^2` cancels the common primitive density exactly as in the
`j=1` case. Comparing `j` with `j=1`, the paired-kernel numerator is therefore
thinned by

```tex
\frac{\rho_j}{\rho_1}=2^{-3(j-1)}.
```

The relay `D` norm is taken over the actual valuation-`j` relay lattice. In the
continuum normalization,

```tex
D_j(w)^2\sim \rho_j\int_S |\zeta|^4|w(\zeta)|^2\,d\zeta.
```

After dualizing in this weighted output space, the squared operator bound has
the same paired kernels `M_{oa}` and `M_{pmid}` as for `j=1`, multiplied by the
density ratio `rho_j/rho_1`. Taking square roots gives

```tex
C_{00j,ord}^*
\leq
\left(\frac{\rho_j}{\rho_1}\right)^{1/2}C_{001,ord}^*
=2^{-3(j-1)/2}C_{001,ord}^*.
```

This completes the continuum density-scaling reduction. The only paper-level
bookkeeping still needed here is to state the finite-k error as an `o(1)` or
`limsup` term. No additional Navier--Stokes cancellation is needed for `j>=2`;
all ordered-slot cancellation has already been absorbed into `C_{001,ord}^*`.

With this density-scaling theorem, the full higher relay tail is geometric:

```tex
\sum_{j\geq2} C_{00j,ord}^*
\leq C_{001,ord}^* \sum_{m\geq1} 2^{-3m/2}
= 0.5469181607\ldots\, C_{001,ord}^*.
```

So the non-primitive relay problem collapses to bounding `C_{001,ord}^*`; the
tail then costs only about `54.7%` of that bound in the worst continuum
estimate. The ordered-slot work below gives the current conservative bound
`C_{001,ord}^* <= 0.5732817578...`.

### Theorem Target: single-output dual formulation

The output-slot first-relay piece can be reduced from a three-field variational
problem to a quadratic operator norm. For primitive valuation-0 input `u`,
define the relay quadratic output `F_{001}(u,u)` by restricting the NS
convolution to triads with valuation pattern `(0,0,1)` and placing the output on
the valuation-1 relay slot, in the convention where the output pairs directly
with the relay field `w`. Then

```tex
B_{001}(u,u,w)=\langle F_{001}(u,u),w\rangle.
```

Taking the supremum over relay fields `w` first gives

```tex
C_{001,out}^*
= \sup_{u\ne0}
\frac{\|F_{001}(u,u)\|_{D^{-1}}}{X(u)^2},
```

where

```tex
\|F\|_{D^{-1}}^2
:= \int_S |\xi|^{-4}|F(\xi)|^2\,d\xi
```

in the rescaled continuum model, after projecting onto the divergence-free relay
subspace. Equivalently, if one defines `Q_{001}:=(-\Delta)^{-1}F_{001}`, then

```tex
B_{001}(u,u,w)=\langle Q_{001}(u,u),\Delta w\rangle,
\qquad \|F_{001}\|_{D^{-1}}=\|Q_{001}\|_{L^2}.
```

The memo uses the `F_{001}` convention below, since it keeps the dual norm
explicit. This is not a Rayleigh quotient, but it is a quadratic-to-linear
operator norm. It is therefore amenable to Schur, Hilbert--Schmidt, or
angular/radial decompositions.

Using the polarization envelope from the existing B-shell kernel, the scalar
amplitude before dualizing has the same geometric factor

```tex
K(\xi,\eta)
= |\zeta|^2\left(1-\frac{(\xi\cdot\eta)^2}{|\xi|^2|\eta|^2}\right),
\qquad \zeta=-(\xi+\eta),
```

but now restricted to the valuation-0/valuation-1 residue classes. The
`D^{-1}` dual norm cancels the output factor `|\zeta|^2`: the weighted output is
`|\zeta|^{-2}F_{001}`. Therefore the squared dual-norm majorant is governed not
by the raw `K_0` Fredholm kernel, but by the safer relay-dual angular kernel

```tex
\left(1-\frac{(\xi\cdot\eta)^2}{|\xi|^2|\eta|^2}\right)^2.
```

This is a majorant for the dualized relay norm, not an eigenvalue formula for
`C_{001,out}^*`.

Writing `s=|\xi|^2`, `t=|\eta|^2`, and `x=\cos\angle(\xi,\eta)`, the radial
`\ell=0` relay-dual kernel is

```tex
L_0(s,t)
:=\int_{x_1(s,t)}^{x_2(s,t)} (1-x^2)^2\,dx,
```

with the same bounds

```tex
x_1(s,t)=\frac{1-s-t}{2\sqrt{st}},
\qquad
x_2(s,t)=\frac{2-s-t}{2\sqrt{st}}.
```

For `s,t\in[1,2)` the clipping constraints are inactive, giving the closed form

```tex
L_0(s,t)
=\frac{
15s^4-60s^3t-90s^3+90s^2t^2+90s^2t+210s^2
-60st^3+90st^2+140st-225s
+15t^4-90t^3+210t^2-225t+93
}{480(st)^{5/2}}.
```

The endpoint proof below shows that `L_0(s,t)/(st)` is maximized at `s=t=1`,
with value `203/480`.

The output-slot proof bounds

```tex
\sup_{X(u)=1}\|F_{001}(u,u)\|_{D^{-1}}.
```

The angularly averaged radial Schur bound uses `L_0`, not the raw `K_0`
spectrum. It is only the output-slot piece; the ordered trilinear constant is
completed by the Lions-paired slot reconciliation below.

### Working proof: dual normalization and the `L_0` endpoint bound

The safe first estimate should be written as a majorant, not as a spectral
identity.

Dual normalization. Let `H_1` denote the valuation-1 relay subspace, with

```tex
D(w)^2=\int_S |\zeta|^4|w(\zeta)|^2\,d\zeta.
```

For fixed primitive input `u`, define `F_{001}(u,u)` as the `L^2`-dual output on
`H_1`, already projected onto the divergence-free relay fibers:

```tex
B_{001}(u,u,w)=\langle F_{001}(u,u),w\rangle_{L^2(S)}.
```

Then Cauchy--Schwarz in the weighted relay Hilbert space gives

```tex
|B_{001}(u,u,w)|
\leq
\left(\int_S |\zeta|^{-4}|F_{001}(u,u)(\zeta)|^2\,d\zeta\right)^{1/2}D(w).
```

The bound is sharp for fixed `u`, with extremizer proportional to
`|\zeta|^{-4}F_{001}` inside the relay subspace. Therefore

```tex
\sup_{w\ne0}\frac{|B_{001}(u,u,w)|}{D(w)}
=\|F_{001}(u,u)\|_{D^{-1}}.
```

Let

```tex
G_{001}(u,u)(\zeta):=|\zeta|^{-2}F_{001}(u,u)(\zeta).
```

Then

```tex
\|F_{001}(u,u)\|_{D^{-1}}=\|G_{001}(u,u)\|_{L^2}.
```

Angular kernel. For a relay triad `\xi+\eta+\zeta=0`, the existing B-shell
polarization envelope gives

```tex
|F_{001}(\zeta)|
\lesssim |\zeta|^2
\left(1-\frac{(\xi\cdot\eta)^2}{|\xi|^2|\eta|^2}\right)|u(\xi)|\,|u(\eta)|
```

at the scalar majorant level. Passing to `G_{001}=|\zeta|^{-2}F_{001}` cancels
the output weight, so the squared angular majorant is `(1-x^2)^2`, where
`x=cos angle(\xi,\eta)`.

The condition `\zeta\in S` is

```tex
1\leq |\xi+\eta|^2=s+t+2\sqrt{st}\,x<2,
```

hence

```tex
x_1(s,t)=\frac{1-s-t}{2\sqrt{st}}
\leq x\leq
x_2(s,t)=\frac{2-s-t}{2\sqrt{st}}.
```

For `s,t\in[1,2)`, these bounds are already inside `[-1,1]`: `x_2\le0<1`,
and `x_1>-1` is equivalent to `(\sqrt{s}-\sqrt{t})^2<1`, which follows from
`\sqrt{s},\sqrt{t}\in[1,\sqrt2)`. Thus no clipping is needed in the dyadic
annulus. The radial angular kernel is exactly

```tex
L_0(s,t)=\int_{x_1(s,t)}^{x_2(s,t)}(1-x^2)^2\,dx.
```

The endpoint constant. Since

```tex
L_0(s,t)=\frac{N(s,t)}{480(st)^{5/2}},
```

where

```tex
N(s,t)=15s^4-60s^3t-90s^3+90s^2t^2+90s^2t+210s^2
-60st^3+90st^2+140st-225s
+15t^4-90t^3+210t^2-225t+93,
```

we have

```tex
\frac{L_0(s,t)}{st}=\frac{N(s,t)}{480(st)^{7/2}}.
```

For positive `s,t`, the sign of `\partial_s(L_0/(st))` is the sign of

```tex
P(s,t):=2s\,\partial_sN(s,t)-7N(s,t).
```

Writing `a=s-1`, `b=t-1`, with `0\leq a,b<1`, direct expansion gives

```tex
P=15a^4+60a^3b+210a^3-270a^2b^2-630a^2b-630a^2
+300ab^3-90ab^2-2140ab-1555a
-105b^4+510b^3-30b^2-1345b-751.
```

The positive terms satisfy

```tex
15a^4+60a^3b+210a^3+300ab^3+510b^3
\leq 585a+510b,
```

so

```tex
P\leq -751-970a-835b<0.
```

Thus `L_0(s,t)/(st)` decreases in `s`; by symmetry it also decreases in `t`.
The maximum occurs at `s=t=1`, where

```tex
\sup_{s,t\in[1,2)}\frac{L_0(s,t)}{st}
=L_0(1,1)
=\int_{-1/2}^0(1-x^2)^2\,dx
=\frac{203}{480}.
```

Coarea/fiber Schur step. The scalar endpoint bound above becomes a bound on the
actual output once the convolution fiber is compared with Lebesgue measure on
the output annulus. Fix `s,t` and write

```tex
r:=|\zeta|^2=s+t+2\sqrt{st}\,x.
```

After quotienting the common rotation, the pair-angle measure has the form
`dx\,d\omega_\zeta`, while the output volume element is

```tex
d\zeta = \frac12\sqrt r\,dr\,d\omega_\zeta
=\sqrt{st}\sqrt r\,dx\,d\omega_\zeta.
```

Thus the pushforward density of pair angles relative to output volume is

```tex
h_{s,t}(r)=\frac{1}{\sqrt{st}\sqrt r}\leq1,
```

because `s,t,r\in[1,2)` on the dyadic annulus. For a scalar fiber integrand
`A(\xi,\eta)`, the convolution output has the form

```tex
T_{s,t}A(\zeta)=h_{s,t}(r)\,\mathbb{E}(A\mid \xi+\eta=-\zeta).
```

Jensen's inequality on each fiber gives

```tex
|T_{s,t}A(\zeta)|^2
\leq h_{s,t}(r)^2\,\mathbb{E}(|A|^2\mid \xi+\eta=-\zeta).
```

Integrating in `d\zeta` and using `h_{s,t}\leq1` yields the contraction

```tex
\|T_{s,t}A\|_{L^2(d\zeta)}^2
\leq \iint_{\xi\in S_s,\eta\in S_t}|A(\xi,\eta)|^2\,d\omega_\xi d\omega_\eta.
```

Applying this with

```tex
A(\xi,\eta)=
\left(1-\frac{(\xi\cdot\eta)^2}{|\xi|^2|\eta|^2}\right)
|u(\xi)|\,|u(\eta)|
```

gives the desired vector/fiber Schur majorant, since the divergence-free relay
projection is an orthogonal projection and cannot increase the fiber norm:

```tex
\|G_{001}(u,u)\|_{L^2}^2
\leq
\iint_{[1,2)^2} L_0(s,t)\,e_u(s)e_u(t)\,d\mu(s)d\mu(t),
```

where `e_u` is the radial energy density of the primitive valuation-0 input and
the normalization is

```tex
X(u)^2=\int_1^2 s\,e_u(s)\,d\mu(s).
```

Equivalently, with `d\alpha(s):=s e_u(s)d\mu(s)/X(u)^2`, the desired right-hand
side is

```tex
X(u)^4
\iint_{[1,2)^2} \frac{L_0(s,t)}{st}\,d\alpha(s)d\alpha(t).
```

Combining with the endpoint calculation gives the continuum single-output relay
bound

```tex
C_{001,out}^* \leq
\left(\sup_{s,t\in[1,2)}\frac{L_0(s,t)}{st}\right)^{1/2}
=
\sqrt{203/480}=0.6503204338\ldots.
```

This is still a majorant, not a sharpness claim, and it applies only to the
output-slot relay convention. The exact ordered trilinear decomposition used for
`B(v,v,\Delta v)` is handled in the next sections; the final conservative
ordered bound is `C_{001,ord}^* <= 0.5732817578...`, which is smaller than the
single-output bound because Lions pairing introduces shell-difference factors.

### Ordered-slot reconciliation

The complete-triad implementation evaluates ordered terms of the form

```tex
B_\ell(r,m)
=-|\ell|^2\operatorname{Im}\left[(m\cdot \widehat v(r))
\left(\widehat v(\ell)^*\cdot\widehat v(m)\right)\right],
\qquad \ell=r+m.
```

Thus a valuation pattern `(0,0,1)` has three possible placements of the relay
mode in this ordered tensor:

```text
relay in ell-slot:  Delta/output relay        handled by F001 above
relay in r-slot:    advecting input relay     not the same kernel
relay in m-slot:    transported input relay   not the same kernel
```

The single-output `F_{001}` bound controls the first line only. The other two
lines are not automatically covered by the `L_0` endpoint calculation, because
the `D(w)` denominator is attached to a relay input rather than to the
`Delta`-slot output. The sections below first derive raw input-slot majorants
and then supersede the raw slot sum by Lions-paired kernels. That pairing is the
ordered-slot reconciliation: output plus advecting input carries a factor
`s-rho`, and the transported-input pair carries a factor `t-s`.

### Working proof: input-slot relay majorants

The ULU cancellation gives one useful warning but not a complete disposal of the
input slots. In the two-shell lemma, the first and third slots are the same
scalar shell. In the implemented tensor this pairs an `r`-slot term with an
`ell`-slot term only on the shell-diagonal subset where those two modes have the
same squared radius. The valuation split is different: a valuation-1 relay mode
and a valuation-0 primitive mode can lie in the same dyadic block with many
different squared radii. Hence the remaining ordered-slot problem is an
off-diagonal shell-pair bound, not just another ULU vanishing statement.

Use the same annulus variables as above. Let `xi` denote the primitive mode in
the `ell` slot, let `eta` denote the signed other primitive input, and let the
relay mode be

```tex
\zeta=\xi+\eta,
\qquad
s=|\xi|^2,
\quad t=|\eta|^2,
\quad \rho=|\zeta|^2=s+t+2\sqrt{st}\,x.
```

The admissible interval for `x` is the same `[x_1(s,t),x_2(s,t)]` as in the
single-output calculation. After taking the supremum over the relay field in
the `D` norm and rewriting the primitive energies with
`d alpha(s)=s e_u(s)d mu(s)/X(u)^2`, the following scalar slot majorants result.

Output slot. This is the bound already proved above:

```tex
M_{out}(s,t)
=\frac{1}{st}\int_{x_1}^{x_2}(1-x^2)^2\,dx,
\qquad
\sup M_{out}=203/480.
```

Advecting input slot. The relay field appears in the factor
`eta . w_zeta`. The `D^{-1}` dual weight contributes `rho^{-2}`, the output
weight contributes `s^2`, and

```tex
\sin^2\angle(\eta,\zeta)
=\frac{s(1-x^2)}{\rho}.
```

After the `X(u)^2` normalization, the scalar majorant is therefore

```tex
M_{adv}(s,t)
=\int_{x_1}^{x_2}\frac{s^2(1-x^2)}{\rho^3}\,dx.
```

Transported input slot. The relay field appears in the inner product with the
`ell`-slot primitive coefficient, while the derivative factor comes from the
relay vector acting on the other primitive input. The corresponding scalar
majorant is

```tex
M_{mid}(s,t)
=\int_{x_1}^{x_2}\frac{s^2(1-x^2)}{\rho^2t}\,dx.
```

These are still majorants; they ignore possible phase cancellation and possible
orthogonality between the three slot outputs. But their endpoint constants are
explicit. Since

```tex
\rho=s+t+2\sqrt{st}\,x,
\qquad
dx=\frac{d\rho}{2\sqrt{st}},
\qquad
1-x^2=\frac{4st-(\rho-s-t)^2}{4st},
```

the integration interval becomes exactly `1<=rho<=2`. Writing `L=log 2`,

```tex
M_{adv}(s,t)
=\frac{\sqrt{s}}{8t^{3/2}}
\left[s+t-L+\frac38\bigl(4st-(s+t)^2\bigr)\right],
```

and

```tex
M_{mid}(s,t)
=\frac{s}{8t^2\sqrt{st}}
\left[-1+2(s+t)L+\frac12\bigl(4st-(s+t)^2\bigr)\right].
```

The derivative signs reduce to endpoint checks on elementary quadratics. On
`1<=s,t<=2`, after multiplying by positive factors,

```tex
\partial_s M_{adv}>0,\qquad \partial_t M_{adv}<0,
\qquad
\partial_s M_{mid}>0,\qquad \partial_t M_{mid}<0.
```

The needed signs follow, for example, from the bounds

```tex
8L+15s^2-18st-24s+3t^2-8t\leq8L-29<0,
```

```tex
24L+9s^2-6st-24s-3t^2-8t\leq24L-32<0,
```

```tex
12Ls+4Lt-5s^2+6st-t^2-2\geq16L-2>0,
```

and

```tex
-20Ls-12Lt+5s^2-6st+t^2+10\leq10-32L<0.
```

Hence both input-slot majorants attain their suprema at the boundary
`s -> 2`, `t=1`. The exact squared constants are

```tex
\sup M_{adv}=\frac{\sqrt2}{64}\bigl(21-\log 256\bigr)
=0.3415065572\ldots,
\qquad
\sup M_{mid}=\frac{3\sqrt2}{16}\bigl(\log 16-1\bigr)
=0.4700285647\ldots.
```

The corresponding operator scales are `0.5843856023...` and
`0.6855862937...`.

The raw input-slot constants are useful diagnostics, but the direct raw-slot
triangle sum is superseded by the Lions-paired recombination below.

### Working proof: Lions-paired slot recombination

The right combination is visible before taking absolute values. Write the
unweighted trilinear form as `tilde B`, so that for a single output shell of
squared radius `n_c`,

```tex
B(a,b,\Delta c)=-n_c\widetilde B(a,b,c).
```

Lions antisymmetry gives

```tex
\widetilde B(a,b,c)=-\widetilde B(c,b,a).
```

Therefore the output-relay and advecting-input-relay terms pair as

```tex
B(\xi,\eta,\Delta\zeta)+B(\zeta,\eta,\Delta\xi)
=(s-\rho)\widetilde B(\xi,\eta,\zeta),
```

instead of carrying separate coefficients `rho` and `s`. This recovers the ULU
vanishing on the shell diagonal `s=rho`, but also gives a quantitative
off-diagonal factor for the valuation split.

Similarly, the transported-input slot pairs across the two primitive outer
slots:

```tex
B(\eta,\zeta,\Delta\xi)+B(\xi,\zeta,\Delta\eta)
=(t-s)\widetilde B(\eta,\zeta,\xi).
```

Thus the cancellation-aware relay problem should be governed not by the raw
`M_out`, `M_adv`, and `M_mid` sum, but by two paired kernels. The first is the
output/advecting pair

```tex
M_{oa}(s,t)
=\frac1{st}\int_{x_1}^{x_2}
\frac{(s-\rho)^2}{\rho^2}(1-x^2)^2\,dx,
```

and the second is the paired transported-input contribution

```tex
M_{pmid}(s,t)
=\int_{x_1}^{x_2}
\frac{(s-t)^2(1-x^2)}{\rho^2t}\,dx.
```

The `rho`-substitution gives a compact closed form for the paired middle term:

```tex
M_{pmid}(s,t)
=\frac{(s-t)^2}{8t^2s\sqrt{st}}
\left[-1+2(s+t)L+\frac12\bigl(4st-(s+t)^2\bigr)\right],
\qquad L=\log2.
```

For the output/advecting pair,

```tex
M_{oa}(s,t)=\frac{P_{oa}(s,t)}{320(st)^{7/2}},
```

where

```tex
P_{oa}=5s^6-20s^5t+30s^4t^2+150s^4-20s^3t^3-80s^3t-300s^3
+5s^2t^4+40s^2t^2-120s^2t+350s^2+40st^3-120st^2+280st
-20s(s-t)^2(3s^2+t^2)L-225s+10t^4-60t^3+140t^2-150t+62.
```

The paired middle endpoint is now elementary. Write `r=s/t`. Then

```tex
M_{pmid}(rt,t)
=\frac{(r-1)^2}{8r^{3/2}}
\left[-\frac1{t^2}+\frac{2(r+1)L}{t}
+2r-\frac{r^2+1}{2}\right].
```

Since `(r+1)L>1` throughout the admissible ratio range, the bracket decreases
as `t` increases. For `r>=1`, the maximum at fixed `r` is therefore at `t=1`.
On `1<=r<=2`, differentiating the resulting one-variable function gives a
positive multiple of

```tex
-\left[(r-1)(5r^2-8r-9)-4L(3r^2+2r+3)\right],
```

which is positive because `(r-1)(5r^2-8r-9)<=0`. Hence the maximum in this
branch occurs at `r=2`. For `r<=1`, the minimum admissible scale is `s=1`; if
`q=1/r`, then

```tex
M_{pmid}(1,q)=q^{-1}M_{pmid}(q,1),
```

so this branch is also bounded by the `q=2` endpoint. Therefore

```tex
\sup M_{pmid}=\frac{3\sqrt2}{64}\bigl(\log16-1\bigr)
=0.1175071412\ldots.
```

For the output/advecting pair, the endpoint can be certified as follows. Define

```tex
D_{oa}(s,t):=-\frac{320(st)^{9/2}}{s}\,\partial_t M_{oa}(s,t).
```

It is enough to show `D_{oa}>0`, since then `M_{oa}` decreases in `t`. Use the
elementary rational enclosure

```tex
0.693<L<0.694.
```

Replacing each positive `L`-monomial in `D_{oa}` by `0.693` and each negative
`L`-monomial by `0.694` gives a lower polynomial `D_-`. With `u=s-1` and
`v=t-1`, the bidegree `(6,4)` Bernstein coefficient matrix for `D_-` on
`0<=u,v<=1` is

```text
[2917/100, 2279/50, 6697/100, 18247/200, 1157/10]
[13523/600, 6813/200, 14629/300, 39353/600, 12487/150]
[8881/750, 52919/3000, 37481/1500, 100391/3000, 15958/375]
[41/8, 4019/500, 581/50, 7757/500, 9629/500]
[21823/1500, 65717/3000, 23137/750, 60601/1500, 36973/750]
[24757/600, 9539/150, 5507/60, 6171/50, 46639/300]
[3889/50, 2509/20, 9433/50, 26263/100, 8536/25]
```

All entries are positive; the minimum is `41/8`. Since Bernstein basis functions
are nonnegative and form a partition of unity, `D_->0`, hence `D_{oa}>0` and
`M_{oa}(s,t)<=M_{oa}(s,1)`.

It remains to bound the edge. Put `a=sqrt{s}`. A direct factorization gives

```tex
\frac{17}{320}-M_{oa}(s,1)
=\frac{(a-1)W(a)}{320a^7},
```

where

```tex
W(a)=60La^9+60La^8-60La^7-60La^6+20La^5+20La^4-20La^3-20La^2
-5a^{11}-5a^{10}+15a^9+15a^8-165a^7-148a^6
+252a^5+252a^4-23a^3-23a^2+2a+2.
```

The coefficient of `L` in `W` factors as

```tex
20a^2(a-1)(a+1)^2(3a^4+1),
```

so it is nonnegative for `1<=a<=sqrt2`. Thus `W(a)` is bounded below by the
polynomial obtained by replacing `L` with `0.693`. With
`a=1+(sqrt2-1)y`, `0<=y<=1`, its Bernstein coefficients are all positive; in
fact they are positive rationals plus positive rational multiples of `sqrt2`
except for the first coefficient `169`. One convenient coefficient vector is

```text
169,
(12094 sqrt2 + 34381)/275,
(114852 + 118662 sqrt2)/1375,
(330739 + 418663 sqrt2)/4125,
202 sqrt2/3 + 1204817/8250,
69418 sqrt2/1155 + 1996831/11550,
76169/1155 + 282279 sqrt2/1925,
495069 sqrt2/5500 + 2643251/16500,
114138 sqrt2/1375 + 755453/4125,
418269/2750 + 312713 sqrt2/2750,
42308/275 + 33381 sqrt2/275,
2109 sqrt2/25 + 5509/25.
```

Therefore `W(a)>0` on the edge, and

```tex
\sup M_{oa}=\frac{17}{320}=0.053125,
```

with equality at `(s,t)=(1,1)`. The paired kernels are structurally correct
because the shell-difference factors are forced by Lions antisymmetry before any
Cauchy--Schwarz estimate.

This already gives a conservative ordered first-relay bound without any
orthogonality claim. If `F_{oa}` and `F_{pmid}` denote the two paired relay
outputs in the `D^{-1}` dual norm, then

```tex
\|F_{oa}+F_{pmid}\|_{D^{-1}}
\leq \|F_{oa}\|_{D^{-1}}+\|F_{pmid}\|_{D^{-1}}.
```

Therefore

```tex
C_{001,ord}^*
\leq
\sqrt{\frac{17}{320}}
+\left(\frac{3\sqrt2}{64}(\log16-1)\right)^{1/2}
=0.5732817578\ldots.
```

This is not the sharp paired value, but it is a rigorous ordered-slot closure if
the coarea/fiber Schur contraction is applied separately to the two paired
operators.

Joint Schur-kernel sharpening. The literal output-space cross term
`<F_oa,F_pmid>_{D^{-1}}` contains cross-fiber terms after convolution, so a
pointwise coefficient-only involution is not a proof. The useful theorem is one
step earlier, at the Hilbert--Schmidt Schur-kernel level.

For fixed triad geometry put

```tex
\xi+\eta+\zeta=0,
\qquad s=|\xi|^2,
\qquad t=|\eta|^2,
\qquad \rho=|\zeta|^2,
\qquad c=\xi\cdot\eta,
\qquad q=1-{c^2\over st}.
```

If `P_zeta` denotes projection onto `zeta^perp`, the two angular bilinear maps
behind the paired kernels are, up to their scalar Lions coefficients,

```tex
A_{oa}(a,b)=(\eta\cdot a)P_\zeta b,
\qquad
A_{pmid}(a,b)=(\zeta\cdot b)P_\zeta a,
\qquad a\in\xi^\perp,
\quad b\in\eta^\perp.
```

Their Hilbert--Schmidt mixed trace is

```tex
T_{HS}(s,t,\rho)
:=\sum_{a,b}\langle A_{oa}(a,b),A_{pmid}(a,b)\rangle
=q{(s+c)(t+c)\over\rho}.
```

On `1<=s,t,rho<=2`, this trace is nonnegative because
`s+c=(rho+s-t)/2>=0` and `t+c=(rho+t-s)/2>=0`. After swapping the primitive
inputs, the scalar Lions coefficients give

```tex
(s-\rho)(t-s)+(t-\rho)(s-t)=-(s-t)^2\leq0.
```

Since `T_HS` is symmetric in `s,t`, the symmetrised mixed Schur kernel is
nonpositive. Thus the Schur contraction may be applied to the already-combined
paired output, giving the pointwise joint scalar kernel

```tex
K(s,t):=M_{oa}(s,t)+M_{pmid}(s,t).
```

The joint endpoint is better than combining the two separate suprema. At
`(s,t)=(2,1)`,

```tex
M_{oa}(2,1)={\sqrt2(103-130\log2)\over1280},
\qquad
M_{pmid}(2,1)={3\sqrt2(4\log2-1)\over64},
```

and hence

```tex
K_*:=K(2,1)={\sqrt2(110\log2+43)\over1280},
\qquad
\sqrt{K_*}=0.3629733751\ldots.
```

Endpoint certificate. With `s=a^2`, `t=b^2`, define

```tex
G(a,b):=1280a^7b^7\bigl(K_*-K(a^2,b^2)\bigr).
```

After the substitution `a=1+(sqrt2-1)x`, `b=1+(sqrt2-1)y`, `G` becomes a
bidegree `(12,10)` polynomial on `[0,1]^2`. Using `0.693<log2<0.694`
coefficientwise, all Bernstein coefficients of the lower polynomial are
nonnegative; the only zero coefficient is the endpoint coefficient
corresponding to `(x,y)=(1,0)`. Therefore `K(s,t)<=K_*` on `[1,2]^2`.
The reproducible audit is `scripts/gap3/route_a_joint_kernel_cert.py`.

The same audit now includes the k=12 finite-cell closure.  With
`h=2^{-12/2}=1/64` and `eps=(sqrt6+3h/4)h`, the thickened square
`[1-eps,2+eps]^2` is contained in the rational enlarged square obtained from
`a,b in [49/50,10/7]` after `s=a^2`, `t=b^2`.  Bernstein positivity on that
enlarged square proves the simple joint bounds

```tex
|\partial_s K|\le {2\over5},
\qquad
|\partial_t K|\le {3\over5},
\qquad
K_{collar}\le {3\over10}.
```

Using `log2<=347/500` in `K_*`, the finite relay envelope satisfies

```tex
U_{12}\le 1.3034641156 < C(I_2)2^6=1.4554793862\ldots,
```

so the certified k=12 margin is at least `0.1520152706` on the rescaled scale.

The same joint Schur framework now has a finite weighted-row replacement for the
k=11 endpoint.  The radial paired-kernel budget has enough slack once the row
inflation is certified directly.  The row certificate gives

```tex
R_{11}\le 1.011760776695,
```

and the integrated audit reports

```text
sqrt(R_11) * U_11,radial <= 0.965192260185
   < C(I_2)2^(11/2) = 1.029179343858.
```

Thus the joint Schur and weighted-row finite certificates close k=11 and k=12.
The remaining finite range after the joint Schur framework alone was k=8--10;
that range is handled by the midpoint-centered theorem below.

### Working theorem: midpoint-centered finite block closure for k=8--10

The missing finite blocks are not closed by the universal joint Schur tower.  They
are closed by using the global cancellation identity before taking absolute
values.  For divergence-free `u`,

```tex
B(u,u,u)=0.
```

Therefore the Laplacian output coefficient can be shifted by any constant
multiple of the identity.  In the rescaled annulus `1<=rho<2`, choose the
midpoint `3/2`.  The raw output factor `rho` is replaced by `rho-3/2`, giving
the centered scalar kernel

```tex
K_c(s,t)=\int_1^2 { (\rho-3/2)^2\,\operatorname{disc}(s,t,\rho)^2
            \over 32(st)^{7/2}\rho^2}\,d\rho,
\qquad
\operatorname{disc}=4st-(\rho-s-t)^2.
```

The audit `scripts/gap3/route_b_centered_kernel_cert.py` proves, by Bernstein
coefficient positivity on the enlarged square `a,b in [1,10/7]`, that

```tex
K_c\le {1\over60},
\qquad
|\partial_s K_c|,|\partial_t K_c|\le {1\over25},
\qquad
K_{c,collar}\le {9\over128}.
```

For a finite block `k`, put `h=2^{-k/2}`.  For relay height `j`, set
`alpha_j=1` for `j=0` and `alpha_j=2^j` for `j>=1`, and set the relay weight
`w_0=1`, `w_j=2^{-3(j-1)/2}` for `j>=1`.  The centered finite-row majorant for
the reduced block `k-2j` is

```tex
\sqrt{R_{k-2j}}
\left({1\over60}
 +h^2\left({1\over25}+\alpha_j^2\left({1\over25}+{18\over128}\right)\right)
\right)^{1/2},
```

where `R_m` is the row discretization guard for the centered kernel.  The
guards used in the mixed certificate are

```tex
R_6,R_8\le {5\over4},
\qquad
R_7,R_9,R_{10}\le {3\over2}.
```

The even guards are interval-checked by
`scripts/gap3/centered_row_mp_verify.py --interval`; the wide-margin top rows
for k=9 and k=10 are checked by the guarded coarse certificate
`scripts/gap3/centered_row_coarse_cert.py` with `eps_g=1e-8` and `eps_k=1e-10`.
The observed/certified worst rows are

```text
k6  R_upper=1.103121376559912  at n=88    (interval)
k8  R_upper=1.028818401493566  at n=438   (interval)
k7  R_upper=1.037908796942949  at n=163   (interval)
k9  R_upper=1.007609447255     at n=512   (guarded coarse)
k10 R_upper=1.004985996223     at n=1968  (guarded coarse)
```

At the deepest relay levels, use the already-certified low-block constants when
they are sharper than the centered row majorant.  The resulting mixed bounds are

```text
k= 8  U_mixed <= 0.340202543023 < C(I_2)2^4        = 0.363869846549
k= 9  U_mixed <= 0.369886147671 < C(I_2)2^(9/2)    = 0.514589671929
k=10  U_mixed <= 0.374439743029 < C(I_2)2^5        = 0.727739693099
```

Equivalently,

```text
C(I_8)  <= 0.021262658939 < C(I_2),
C(I_9)  <= 0.016346812706 < C(I_2),
C(I_10) <= 0.011701241970 < C(I_2).
```

This proves the needed finite closure for k=8--10 without claiming the exact
optimizer values in those blocks.

With the same higher-relay multiplier, the conservative triangle relay tower is

```tex
0.5732817578\ldots\times1.5469181607\ldots
=0.8868199623\ldots,
```

while the joint Schur-kernel refinement gives

```tex
0.3629733751\ldots\times1.5469181607\ldots
=0.5614901059\ldots.
```

Thus the joint Schur theorem is a genuine sharpening, not a prerequisite for a usable
ordered-slot relay bound.

Scale check. With the conservative paired ordered-slot bound, the whole
first-relay plus higher-relay tower would satisfy

```tex
C_{001,ord}^*\left(1+\sum_{m\geq1}2^{-3m/2}\right)
\leq 0.5732817578\ldots\times1.5469181607\ldots
=0.8868199623\ldots.
```

The comparison threshold is `C(I_2)2^{k/2}`. Numerically,

```text
k=11: 1.0291793439
k=12: 1.4554793862
k=13: 2.0583586877
```

These numbers explain why the replacement theorem can be much weaker than the
old plateau claim.  The final finite proof does not rely on a universal joint
Schur bound for k=8--10: those blocks are closed by the centered finite theorem
above.  For k=11 and k=12, the joint Schur and weighted-row estimates give the
direct finite certificates just stated.
Lemma 6 below formalizes the large-k decomposition used for the tail theorem.

### Lemma 6 Target: primitive-core family envelope

The ordered relay tower is now separated from the primitive core. Let

```tex
R_{ord}:=0.5732817578\ldots
\left(1+\sum_{m\geq1}2^{-3m/2}\right)
=0.8868199623\ldots.
```

For each primitive raw feature signature `sigma`, define the same-valuation core
envelope

```tex
P_\sigma^{(k)}
:=2^{k/2}\sup_{u\in\mathcal P_\sigma(k)}
\frac{|B_{0,0,0}(u,u,\Delta u)|}{X(u)^2D(u)},
```

where `\mathcal P_\sigma(k)` is the class of valuation-0 fields in `I_k` whose
active primitive shells have raw feature signature `sigma`, after removing only
the same-shell vanishing pieces; ULU remains unless it is separately controlled.
The family envelope should take the form

```tex
E_\sigma(k)\leq P_\sigma^{(k)}+R_{ord}+o(1).
```

Thus the remaining quantitative theorem is

```tex
\limsup_{k\to\infty}P_\sigma^{(k)}\leq P_\sigma^*,
```

with `P_sigma^*` below the headroom left by `R_ord` after the finite exceptional
blocks. This is the formal version of the hypothesis:

```text
global spikes = max over primitive family envelopes plus a universal ordered relay tower.
```

Working proof: primitive-core paired majorant. The same Lions-paired annulus
kernels used for the ordered first-relay estimate give a conservative universal
bound for the same-valuation primitive core. The key point is that the endpoint
proof of `M_oa` and `M_pmid` did not use relay sparsity after passing to the
nonnegative coarea/Schur majorants. It integrated over the full annulus geometry

```tex
1\leq s,t,\rho<2,
\qquad \rho=s+t+2\sqrt{st}\,x,
```

and only then noted that residue restrictions can thin the admissible lattice.
Therefore the same majorant applies to an unrestricted primitive same-valuation
ordered triad family; imposing a raw feature signature `sigma` can only reduce
the admissible set.

Let `a,b,c` be independent valuation-0 fields in the rescaled annulus, with
`c` in the Laplacian/D slot. After removing the same-shell vanishing pieces and
retaining ULU as an ordinary off-diagonal contribution, Lions antisymmetry pairs
the ordered tensor into the same two outputs as before:

```tex
F_{oa}:\quad (s-\rho)\widetilde B,\qquad
F_{pmid}:\quad (t-s)\widetilde B.
```

The coarea/fiber Schur contraction and the endpoint calculations already proved

```tex
\|F_{oa}\|_{D^{-1}}
\leq \left({17\over 320}\right)^{1/2}X(a)X(b),
```

and

```tex
\|F_{pmid}\|_{D^{-1}}
\leq
\left({3\sqrt2\over64}(\log 16-1)\right)^{1/2}X(a)X(b).
```

Taking the triangle inequality in the dual output norm gives the universal
primitive-core operator bound

```tex
C_{000,pair}^*
\leq
\left({17\over 320}\right)^{1/2}
+
\left({3\sqrt2\over64}(\log 16-1)\right)^{1/2}
=0.5732817578\ldots.
```

Setting `a=b=c=u` gives the desired cubic bound. Hence for every primitive raw
feature signature `sigma`,

```tex
\limsup_{k\to\infty}P_\sigma^{(k)}
\leq P_\sigma^*,
\qquad
P_\sigma^*:=C_{000,pair}^*
\leq0.5732817578\ldots.
```

The finite-k error is the same residue/Riemann-sum `o(1)` already used in the
relay theorem. This is deliberately not claimed to be sharp: it discards
signature-specific sparsity, phase cancellation, and possible orthogonality
between the paired outputs. But it closes the required primitive-core limsup at
a conservative universal level.

Working proof of the envelope decomposition. Decompose a field by vector
valuation,

```tex
u=\sum_{a\geq0}u_a,
\qquad u_a:=\mathbf 1_{\nu_2(p)=a}u.
```

By Lemma 3, every nonzero triad has valuation pattern `(a,a,a)` or, after
permutation, `(a,a,a+j)` with `j>=1`.

For `(a,a,a)`, divide all three modes by `2^a`. The ordered triad type is
unchanged because the raw features are stable under `n -> 4n`, and the rescaled
constant is exactly the primitive-core constant in the lower block. Thus, for a
family `sigma`,

```tex
|B_{a,a,a}(u_a,u_a,\Delta u_a)|
\leq 2^{-k/2}P_\sigma^{(k-2a)}X(u_a)^2D(u_a).
```

For `(a,a,a+j)`, the same division by `2^a` gives a primitive ordered relay of
height `j`; the relay theorem gives

```tex
|B_{a,a,a+j}|
\leq 2^{-k/2}C_{00j,ord}^*X(u_a)^2D(u_{a+j}).
```

Summing over `a` is harmless because the valuation projections are orthogonal
for `X` and `D`:

```tex
\sum_a X(u_a)^2D(u_a)\leq X(u)^2D(u),
\qquad
\sum_a X(u_a)^2D(u_{a+j})\leq X(u)^2D(u).
```

Taking the supremum over `a` in the primitive term and summing the relay
constants in `j` gives

```tex
2^{k/2}R(u)
\leq \sup_a P_\sigma^{(k-2a)}
+ C_{001,ord}^*\left(1+\sum_{m\geq1}2^{-3m/2}\right)+o(1).
```

The `o(1)` is only the continuum/residue Riemann-sum error in the relay theorem;
the valuation decomposition and exact descendant scaling are algebraic. This is
the promised reduction from the full family envelope to the primitive-core
limsup plus the universal relay tower `R_ord`.

## Proof Status After Primitive-Core Closure

Lemmas 1--5 give the arithmetic decomposition into a same-valuation primitive
core plus valuation-restricted ordered relay constants `C_{00j,ord}`. The
density-scaling theorem reduces the non-primitive relay problem to the ordered
first-relay constant `C_{001,ord}^*`. The dual normalization, coarea/fiber Schur
step, and scalar `L_0` endpoint bound are isolated above for the single-output
continuum relay.
The ordered input slots have also been reduced to explicit scalar majorants
`M_adv` and `M_mid`, with endpoint suprema proved by the `rho`-variable
calculus. Lions pairing further reduces the structurally correct ordered problem
to the smaller kernels `M_oa` and `M_pmid`; both endpoint suprema are now
proved. A conservative triangle combination gives a usable ordered first-relay
bound; orthogonality is now only a possible sharpening.
The required primitive-core limsup gap is now closed at a conservative universal
level:

```text
P_sigma^* <= C_{000,pair}^* <= 0.5732817578... for every raw feature-signature family.
```

The paired-output orthogonality question is not part of the required limsup
closure. It is only an optional sharpening of the already-closed ordered relay
tower and, if needed, the primitive-core paired majorant.

The existing B-shell variational framework already gives the qualitative decay
shape

```tex
C(I_k) \leq C_B^* 2^{-k/2}
```

with `C_B^* < infinity`. The superseded part was the sharp-looking numerical
claim `C_B^* <= 0.157`, not the scaling shape itself. The taxonomy repair should
therefore be viewed as a quantitative decomposition of `C_B^*` into smaller
family constants:

```tex
C_B^* \leq \sup_\sigma C_\sigma^*,
```

where `sigma` ranges over primitive raw feature signatures and their allowed
relay graphs. The target is to show every relevant `C_sigma^*` is below the
rescaled threshold needed after the finite exceptional blocks.

The exact gap is this:

1. Lemma 2 controls exact dyadic descendants of a known optimizer.
2. Lemma 3 says every triad has at least two modes at the same minimum 2-adic
   scale.
3. Lemma 5 identifies the relay as a lower-density, lower-primitive-block object.
4. Valuation-density scaling reduces all `j>=2` ordered relay layers to a
   geometric tail controlled by `C_{001,ord}^*`, by the standard
   residue-restricted Riemann-sum lemma stated above.
5. The single-output dual normalization reduces `C_{001,out}^*` to
   `sup ||F_{001}(u,u)||_{D^{-1}}/X(u)^2`.
6. The coarea/fiber Schur step and scalar angular endpoint bound give the
   single-output continuum relay majorant `sqrt(203/480)`.
7. The raw input slots reduce to `M_adv` and `M_mid`; their individual suprema
   are explicit.
8. Lions pairing replaces the raw slot sum by the smaller paired kernels
   `M_oa` and `M_pmid`; both endpoints are proved, and triangle combination
   gives `C_{001,ord}^* <= 0.5732817578...`.
9. The same-valuation primitive core `(0,0,0)` is bounded by the same
   Lions-paired annulus majorant as the ordered first relay:
   `P_sigma^* <= 0.5732817578...` for every raw feature-signature family.

A useful paper theorem is therefore the primitive-core family envelope stated in
Lemma 6 Target with the conservative value above, followed by the effective
finite-k tail certificate.  This is no longer needed to repair any finite hole
below k=13; it is the mechanism that replaces the obsolete plateau assertion for
the infinite tail.

The A/B/C/D taxonomy remains the explanatory structure for spikes and drops and
for any later sharpening, but the current proof package uses conservative
universal paired majorants where they are sufficient.

In other words, the taxonomy is not merely a label for shells. It should tell us
which primitive resonant graphs have large same-valuation core constants, while
Lemma 3 and the ordered relay theorem prevent arbitrary one-off high-scale
relays from escaping that graph structure.

## Replacement Tail Theorem Target

The finite block work now reaches through k=12: k=8--10 are closed by the
midpoint-centered certificate and k=11--12 by the joint Schur/weighted-row
certificates.  Thus `K0=12`,
and the large-k theorem only needs:

```tex
C(I_k) < C(I_2) \quad\text{for all } k \ge 13.
```

Equivalently, since `C(I_k)=2^{-k/2} C_res(k)`, it is enough to prove

```tex
C_res(k) < C(I_2) 2^{k/2}.
```

The right-hand side grows quickly; this is why the replacement theorem is much
weaker than the obsolete 0.157 plateau target.

A practical theorem can therefore be phrased in terms of headroom after the
ordered relay tower. The conservative relay tower costs `0.8868199623...` on the
rescaled scale. Hence the primitive-core envelope only has to fit below

```tex
H(k):=C(I_2)2^{k/2}-0.8868199623\ldots.
```

Numerically the first tail threshold is
`C(I_2)2^(13/2)=2.0583586877...`.  The conservative primitive-core majorant is

```tex
C_{000,pair}^*\leq0.5732817578\ldots,
```

so the fully conservative sum is

```tex
C_{000,pair}^*+R_{ord}
\leq1.4601017200\ldots.
```

This is comfortably below the k=13 threshold, leaving `0.5982569677...` of
k=13-scale margin at the continuum/limsup level.  The effective certificate in
Lemma 7 below supplies the finite-k Riemann-sum error control and closes every
block `k>=13` directly.

### Lemma 7 Target: effective finite-k Riemann-sum closure

The remaining passage from limsup to a finite tail is an effective Riemann-sum
bound. This can be stated without changing the continuum constants.

Use the mesh notation

```tex
h_m:=2^{-m/2}.
```

The primitive core lives on the mesh `h_k`. A relay of height `j` is better
viewed in variables `(p,h)`, where

```tex
p+q+2^jh=0,
```

with `p,q` in `I_k` and the primitive relay representative `h` in `I_{k-2j}`.
After rescaling `p` by `2^{k/2}` and `h` by `2^{(k-2j)/2}`, both variables live
in the same fixed annulus, but their meshes are different:

```tex
h_k,
\qquad h_{k-2j}=2^j h_k.
```

This anisotropic scaling is the important finite-k bookkeeping. A fixed-modulus
residue Riemann sum only sees `O(h_k)` error for the primitive input variable,
while the relay representative contributes `O(2^j h_k)`. The relay density
factor already proved in the continuum theorem contributes `2^{-3j/2}` at the
operator scale. Hence the finite relay error at height `j` has the summable
shape

```tex
O\left(2^{-3j/2}\,2^j h_k\right)
=O\left(2^{-j/2}h_k\right).
```

Therefore the sum over all relay heights remains finite:

```tex
\sum_{j\geq2} O(2^{-j/2}h_k)=O(h_k).
```

The effective form needed for the paper is the following standard Lipschitz
Riemann-sum lemma. Let `Omega` be a bounded semialgebraic subset of `R^d` whose
boundary has finite Minkowski content, and let `Phi>=0` be `C^1` on a fixed
neighborhood of `Omega`. For an anisotropic rectangular mesh with largest mesh
width `delta`, and for any fixed residue class system, there is a computable
constant `A(Omega,Phi)` such that the residue-restricted upper sum satisfies

```tex
\mathrm{Sum}_{\delta,A}(\Phi;\Omega)
\leq { |A|\over M^d}\int_\Omega \Phi
+A(\Omega,\Phi)\,\delta.
```

Here `A(Omega,Phi)` depends only on the `C^1` norm of `Phi` on the neighborhood
and on the boundary Minkowski content of `Omega`; it does not depend on `k`.
All domains in the primitive-core and relay estimates are finite Boolean
combinations of quadratic annulus constraints, so they are semialgebraic with
finite boundary content. The paired kernels are smooth on these domains because
`rho` is restricted to `[1,2]`.

Applying this lemma to the primitive core, the first relay, and the higher
relay sum gives computable constants `A_{000}`, `A_{001}`, and `A_{00,tail}`
such that

```tex
E_\sigma(k)
\leq C_{000,pair}^*+R_{ord}
+A_{eff}2^{-k/2},
```

where

```tex
A_{eff}:=A_{000}+A_{001}+A_{00,tail}.
```

Thus the conservative finite-k target is

```tex
1.4601017200\ldots + A_{eff}2^{-k/2}
< C(I_2)2^{k/2}.
```

At `k=13`, the available margin is `0.5982569677...`; since `2^{13/2}=90.509...`,
the finite theorem would start at `k=13` from a purely linearized error bound if

```tex
A_{eff}<54.148\ldots.
```

If the explicit `A_eff` obtained from the Lipschitz-bound bookkeeping is larger,
the same theorem still closes after the computable cutoff determined by the last
display. In fact the fully linear anisotropic error is just slightly too lossy
at `k=13`, while the sharper square-root perturbation form below closes `k=13`
directly.

Effective interval certificate for `k>=13`. Put

```tex
h_{13}=2^{-13/2}=0.0110485434560\ldots.
```

A cubical upper-sum cell of side `h_k` has Euclidean radius
`eps_k=sqrt3 h_k/2`. If `1<=|x|^2<=2` and `y` lies in such a cell, then

```tex
\bigl||y|^2-|x|^2\bigr|
\leq 2\sqrt2\,eps_k+eps_k^2
=\left(\sqrt6+{3\over4}h_k\right)h_k.
```

At `k=13` this is `0.0271548467...`, so the rational thickening
`0.97<=s,t,r<=2.03` covers every cubical upper cell for all `k>=13`.
Using `0.693<log2<0.694`, Bernstein coefficient bounds on this thickened square
give the componentwise derivative certificate

```text
|partial_s M_oa|     <= 0.3738915499,
|partial_t M_oa|     <= 0.1385880524,
|partial_s M_pmid|   <= 0.2032395486,
|partial_t M_pmid|   <= 0.5179200717.
```

The rho-boundary collars are certified separately by Bernstein bounds for the
endpoint integrands on `r in [0.97,1] union [2,2.03]`:

```text
K_oa   <= 0.1872458438,
K_pmid <= 0.2303401366.
```

Let

```tex
m_{oa}:={17\over320},
\qquad
m_{pm}:={3\sqrt2\over64}(\log16-1),
\qquad
c_{13}:=\sqrt6+{3\over4}h_{13}.
```

For a relay anisotropy `alpha` define

```tex
B_{oa}(\alpha):=c_{13}
\left(0.3738915499+\alpha(0.1385880524+2\cdot0.1872458438)\right),
```

and

```tex
B_{pm}(\alpha):=c_{13}
\left(0.2032395486+\alpha(0.5179200717+2\cdot0.2303401366)\right).
```

Here `alpha=1` for the primitive core and `alpha=2^j` for the relay of height
`j`, because the relay representative has mesh `h_{k-2j}=2^j h_k`. The certified
finite-k paired majorant is therefore

```tex
P(\alpha,h):=
\sqrt{m_{oa}+B_{oa}(\alpha)h}
+\sqrt{m_{pm}+B_{pm}(\alpha)h}.
```

The conservative finite upper envelope is

```tex
U(k):=P(1,h_k)+\sum_{j\geq1}2^{-3(j-1)/2}P(2^j,h_k).
```

For `k=13`, summing through `j=24` gives

```text
P(1,h_13)+sum_{1<=j<=24} 2^{-3(j-1)/2} P(2^j,h_13)
= 1.9105778717...
```

The remaining tail is bounded by

```tex
2^{3/2}C_{tail}2^{-24}<1.13\cdot10^{-7},
\qquad C_{tail}=0.6646492613\ldots,
```

so

```tex
U(13)<1.910577984.
```

Since

```tex
C(I_2)2^{13/2}=2.0583586877\ldots,
```

the certified margin at the first tail block is

```tex
C(I_2)2^{13/2}-U(13)>0.14778070.
```

For `k>13`, `h_k` decreases while the comparison threshold
`C(I_2)2^{k/2}` increases, so the same certificate proves the conservative tail
bound for every `k>=13`. Thus the finite-tail bottleneck is closed at `k=13`;
the direct finite range `k<=12` is closed by the finite certificates recorded
above.

For audit purposes, if the square-root perturbation is linearized into a single
anisotropic error coefficient, the same certified constants give

```tex
A_{eff,lin}=55.8117851\ldots,
```

which misses the `k=13` linear threshold by only `1.6637456...` and closes from
`k=14`. The square-root form above is the correct way to use the certificate at
`k=13`; the earlier reduced-kernel diagnostic `A_eff=33.237...` should not be
cited as the final finite-k proof because it omits the relay anisotropy factor.

## Why This Explains Spikes And Drops

The old model tried to put all blocks onto one plateau. The data contradict that:
k=6 and k=7 have large mixed webs, and the current dense k=8 incumbent has a
rescaled value near 0.330 while still remaining below the raw k=2 value.

The taxonomy model explains this without panic:

1. A block can spike when a new primitive feature signature appears with favorable
   relay geometry.
2. The same signature then decays along its dyadic descendants.
3. Blocks with different bottom-shell parity or different primitive A/B/C/D mix are
   not required to sit on the same monotone curve.
4. Overlap shells are the likely exceptional cases and should be isolated as finite
   signatures, not averaged into the generic family.

## Immediate Next Mathematical Tasks

1. **DONE (May 27 2026): tail certificate.** Promote the finite-k certificate
   into paper-ready lemma form and attach a reproducibility script for the
   derivative and endpoint-collar bounds.  Script: `scripts/gap3/tail_cert.py`.
   The certified result is `U(13)<1.910577984 < 2.0583586877`, with margin
   `>0.14778070`, and monotonicity then gives all `k>=13`.
2. **DONE (May 30 2026): midpoint-centered finite closure for k=8--10.** The midpoint-
   centered kernel and row guards give
   `U_mixed(8)<=0.340202543023`, `U_mixed(9)<=0.369886147671`, and
   `U_mixed(10)<=0.374439743029`, each below the corresponding
   `C(I_2)2^{k/2}` threshold.  Remaining work: turn the theorem block above
   into polished paper prose and ensure the row-certificate commands are listed
   in the audit appendix.
3. **DONE (May 30 2026): joint Schur finite closure for k=11--12.** The joint
   Schur-kernel endpoint certificate gives `U_12<=1.3034641156`, and the
   weighted-row certificate gives
   `sqrt(R_11) U_11,radial <= 0.965192260185`.  Remaining work: promote both to
   formal lemmas with hypotheses and constants aligned with the scripts.
4. Make the primitive-core paired majorant paper-ready, including the exact
   statement that it is universal in `sigma` and deliberately conservative.
5. Use the primitive feature-signature definition above to make the family
   envelope decomposition `E_sigma(k)` paper-ready.  This is now explanatory and
   structural for the replacement theorem, not a finite-hole repair mechanism.
6. Prepare the final theorem assembly: k=1--7 finite certificates, k=8--10
   midpoint-centered certificate, k=11--12 joint Schur/weighted-row certificate,
   and k>=13 tail certificate.  This should become the
   scaffold for the eventual paper replacement of the obsolete plateau lemma.
7. Keep the k=6, k=7, and k=8 checkpoint diagnostics as qualitative motivation
   for the taxonomy model.  They are not proof inputs for the centered upper
   bounds except where an explicit finite certificate says so.
8. Use `references/SMALL_K_STRUCTURAL_ATLAS.md` as the small-block background
   note.  It should support exposition, not carry the final theorem by itself.

### GPU scan infrastructure (separate future work)

The full-block dense GPU Adam scan for k=8 alone requires several days of
compute with the current pipeline.  Based on the results of the analytic tail
theorem, planned future work includes optimising the scan pipeline - chunked triad
cache, better warm-start strategies, finer incumbent-aware screen thresholds - to
make numerical full-block certification of k=8-12 tractable as an independent
cross-check.  This is a numerical engineering task separate from the analytic
closure above and does not affect the current proof chain.

Joint Schur sharpening status: the useful statement is the Schur-level joint kernel,
not literal output-space paired-output orthogonality.  It improves the relay
tower from `0.88682...` to `0.56149...`.  Further Fourier cancellation may be a
future sharpening, but it is no longer needed for k=8--10 closure; the old
separate-sup orthogonality target `0.63899...` is no longer the best formulation.

## Paper Integration Rule

Do not edit `paper2/ns_cancellation.tex` until the lemmas above are written out
and checked. The eventual paper replacement should be a theorem replacing the
superseded `lem:plateau`, not another numerical plateau assertion.
