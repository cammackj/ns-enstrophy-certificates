# Case-A k=13 Arithmetic Decoupling Certificate Ledger

Status: certificate ledger for Paper 2. This records the arithmetic and finite
residual certificate behind the manuscript claim that the Case-A block k=13 has
relay enhancement rho(13)=1.

Primary manuscript location: `paper2/ns_cancellation.tex`, Theorem
`Case-A odd-k decoupling`.

Primary provenance:

- `references/NS_CANCELLATION.md`, sections 19.10.17 and 19.10.18.
- `scripts/gap3/gap3_arithmetic_decoupling.py`.

## Statement Supported

For the Case-A odd block k=13, the nucleus shells are

```text
n_D = 8192
n_C = 8282
n_A = 16346
```

with `a = 64`. The paper records that all relay shells decouple from the
nucleus triad, so

```text
rho(13) = 1
C(I_13) = C_nuc(13).
```

## Arithmetic Core

The D shell is representation-minimal:

```text
8192 = 64^2 + 64^2 + 0^2
```

up to coordinate permutations and sign flips. Thus every D-shell vector is a
permutation of `(+-64, +-64, 0)`. For a relay mode `p` coupled through a D-shell
mode `q_D`, the triad condition with a nucleus shell `n_nuc` gives

```text
p . q_D = (n_nuc - n_D - n_r) / 2.
```

Since every nonzero component of `q_D` is divisible by 64, integer coupling
through the D shell requires

```text
n_r == n_nuc - n_D  (mod 128).
```

For both the C and A nucleus paths at k=13 this residue is 90 mod 128:

```text
n_C - n_D = 90
n_A - n_D = 8154 == 90 (mod 128).
```

The first relay shell above `n_D` satisfying this congruence is `8410`; the
dominant relay window checked in the original ISC computation ends below this.
Therefore the D-shell contribution to the first relay variation vanishes in the
dominant window by arithmetic divisibility, not by numerical cancellation.

## Parity And D+D Closure

The Case-A nucleus shells at k=13 are all even. Consequently any relay shell
reached by a triad with two nucleus legs must also have even shell number:

```text
n_r = |q + r|^2 = |q|^2 + |r|^2 + 2 q.r
```

is even whenever `|q|^2` and `|r|^2` are even. Thus all odd relay shells have
zero first variation exactly.

For D+D pairs, the minimal D-shell orbit forces

```text
|q + r|^2 in {0, n_D, 2 n_D, 3 n_D, 4 n_D}.
```

Inside the dyadic block `[n_D, 2 n_D)`, the only such shell is `n_D` itself,
which is part of the nucleus rather than a relay shell. Hence D+D relay output
is also exactly absent.

## Remaining k=13 Finite Check

After the analytic parity and D-shell arithmetic eliminations, the remaining
possible Case-A relay channels at k=13 are even C/A-shell couplings. The
recorded finite residual certificate in the manuscript gives

```text
max |d_1| = 1.4e-12
```

for those residual first-variation channels, at roundoff scale and with no
positive first variation. This is the finite residual input used to close the
k=13 Case-A relay-enhancement statement in the manuscript.

The older provenance note records the shell-8224 dump: 96 mode-polarisation
pairs were exact zeros and the remaining near-zero entries were at floating
noise scale from sub-dominant paths, with no significant positive relay
gradient.

## Reproduction

The arithmetic classification can be reproduced with:

```powershell
python scripts/gap3/gap3_arithmetic_decoupling.py
```

This script reports k=13 as Case A with `gap_above = 218` and
`dominant_width = 90`, so the first arithmetically allowed non-nucleus relay
above the D shell lies outside the dominant relay window.
