#!/usr/bin/env sage
"""Sage/Singular elimination probes for the reduced k=3 kernel.

The Python probe in k3_radical_elimination_probe.py builds the symbolic
stationarity system and runs PSLQ.  This Sage script is the next layer: it
uses Singular through Sage to test elimination complexity, especially modulo
finite primes where the natural square roots sqrt(2), sqrt(5), sqrt(7) can be
specialized.

Default mode is a modular full elimination attempt for a single prime.  This
does not prove a characteristic-zero formula, but it is a good complexity
thermometer: if the elimination polynomial for W=C3^2 is already high-degree
mod p, then a pleasant radical expression is unlikely.
"""

from __future__ import annotations

import argparse
import json

from sage.all import GF, Matrix, PolynomialRing, alarm, cancel_alarm, cputime, prime_range, vector
from sage.all import AlarmInterrupt


def quadratic_residue_prime(start: int = 1009, stop: int = 20000, radicands=(2, 5, 7)) -> int:
    for prime in prime_range(start, stop):
        field = GF(prime)
        if all(field(n).is_square() for n in radicands):
            return int(prime)
    raise RuntimeError(f"No prime found in [{start}, {stop}) where {radicands} are squares")


def modular_roots(field, radicands, signs=None):
    signs = signs or {n: 1 for n in radicands}
    return {n: field(signs.get(n, 1)) * field(n).sqrt() for n in radicands}


def parse_root_signs(text: str | None, radicands=(2, 5, 7, 13)):
    if text is None:
        return {n: 1 for n in radicands}
    cleaned = text.strip().replace(",", "")
    if len(cleaned) != len(radicands) or any(char not in "+-" for char in cleaned):
        raise ValueError(f"root signs must be {len(radicands)} characters from '+-' in radicand order {radicands}")
    return {n: (1 if char == "+" else -1) for n, char in zip(radicands, cleaned)}


def build_modular_system(prime: int):
    field = GF(prime)
    roots = modular_roots(field, (2, 5, 7))
    rt2 = roots[2]
    rt5 = roots[5]
    rt7 = roots[7]
    rt10 = rt2 * rt5
    rt35 = rt5 * rt7
    rt70 = rt2 * rt5 * rt7

    ring = PolynomialRing(
        field,
        ("P", "Q", "Rr", "H", "C", "S", "A", "B", "E", "M", "W"),
        order="lex",
    )
    P, Q, Rr, H, C, S, A, B, E, M, W = ring.gens()

    area = 2 + 5 * P**2 + 7 * Q**2 + 5 * Rr**2 + 7 * H**2
    diss = 8 + 25 * P**2 + 49 * Q**2 + 25 * Rr**2 + 49 * H**2

    def p_sigma(sigma):
        return (
            1259
            - 108 * sigma * rt35
            + 637 * Rr**2
            + 2
            * Rr
            * ((162 * sigma * rt7 - 299 * rt5) * C - (180 * rt2 + 39 * sigma * rt70) * S)
        )

    q_theta = 575 + 325 * (C**2 - S**2) - 300 * rt10 * C * S
    p_plus = p_sigma(1)
    p_minus = p_sigma(-1)
    bracket = P * (A + B) + rt2 * H * Rr * E

    equations = [
        C**2 + S**2 - 1,
        A**2 - p_plus,
        B**2 - p_minus,
        E**2 - q_theta,
        M - bracket,
        280**2 * W * area**2 * diss - 70 * Q**2 * M**2,
        (A + B) * area * diss - M * P * (10 * diss + 25 * area),
        area * diss - Q**2 * (14 * diss + 49 * area),
        rt2 * Rr * E * area * diss - M * H * (14 * diss + 49 * area),
    ]

    p_plus_r = p_plus.derivative(Rr)
    p_minus_r = p_minus.derivative(Rr)
    m_r_cleared = P * (p_plus_r * B + p_minus_r * A) + 2 * rt2 * H * E * A * B
    equations.append(m_r_cleared * area * diss - 2 * A * B * M * Rr * (10 * diss + 25 * area))

    p_plus_theta = -S * p_plus.derivative(C) + C * p_plus.derivative(S)
    p_minus_theta = -S * p_minus.derivative(C) + C * p_minus.derivative(S)
    qtheta_theta = -S * q_theta.derivative(C) + C * q_theta.derivative(S)
    equations.append(P * E * (p_plus_theta * B + p_minus_theta * A) + rt2 * H * Rr * A * B * qtheta_theta)

    return ring, equations, (rt2, rt5, rt7)


def build_equivariant_modular_system(prime: int, root_signs=None):
    """Build the smaller 8-variable coordinate-equivariant polynomial system.

    Phi is the compressed cubic numerator from k3_kernel_compress.py:

        Phi = (-455/8) B_eq.

    Since X2=8A and D2=16G, the critical value relation is

        Phi^2 = 1820^2 W A^2 G,  W=C3^2.
    """
    field = GF(prime)
    roots = modular_roots(field, (2, 5, 7, 13), root_signs)
    rt2 = roots[2]
    rt5 = roots[5]
    rt7 = roots[7]
    rt13 = roots[13]

    def rt(n):
        factors = {
            26: rt2 * rt13,
            65: rt5 * rt13,
            91: rt7 * rt13,
            130: rt2 * rt5 * rt13,
            182: rt2 * rt7 * rt13,
            455: rt5 * rt7 * rt13,
            910: rt2 * rt5 * rt7 * rt13,
        }
        return factors[n]

    ring = PolynomialRing(
        field,
        ("x1", "y1", "x3", "y3", "x5", "y5", "x7", "y7", "W"),
        order="lex",
    )
    x1, y1, x3, y3, x5, y5, x7, y7, W = ring.gens()

    def add(z1, z2):
        return z1[0] + z2[0], z1[1] + z2[1]

    def mul(z1, z2):
        return z1[0] * z2[0] - z1[1] * z2[1], z1[0] * z2[1] + z1[1] * z2[0]

    def conj(z):
        return z[0], -z[1]

    def scale(coef, z):
        real, imag = coef
        return real * z[0] - imag * z[1]

    u = (x1, y1)
    v = (x3, y3)
    ww = (x5, y5)
    rr = (x7, y7)

    phi = ring(0)
    # uv terms: [u*v, u*conj(v)]
    phi += scale((-33 * rt(910) + field(525) * rt(26) / 2, 56 * rt(65) + 90 * rt(91)), mul(u, v))
    phi += scale((-field(525) * rt(26) / 2 - 33 * rt(910), 90 * rt(91) - 56 * rt(65)), mul(u, conj(v)))

    # uvw terms: [u*v*w, u*v*conj(w), u*conj(v)*w, conj(u)*v*w]
    phi += scale(
        (-field(21) * rt(130) / 2 + 90 * rt(182), -27 * rt(455) - 70 * rt13),
        mul(mul(u, v), conj(ww)),
    )
    phi += scale(
        (field(21) * rt(130) / 2 + 90 * rt(182), -70 * rt13 + 27 * rt(455)),
        mul(mul(conj(u), v), ww),
    )

    # vwr terms: [v*w*r, v*w*conj(r), v*conj(w)*r, conj(v)*w*r]
    phi += scale(
        (23 * rt(455) + 189 * rt13, -69 * rt(182) + field(63) * rt(130) / 2),
        mul(mul(v, ww), rr),
    )
    phi += scale(
        (-23 * rt(455) + 189 * rt13, -69 * rt(182) - field(63) * rt(130) / 2),
        mul(mul(v, conj(ww)), rr),
    )

    A = 5 * x1**2 + 7 * x3**2 + 5 * x5**2 + 7 * x7**2 + 5 * y1**2 + 7 * y3**2 + 5 * y5**2 + 7 * y7**2 + 2
    G = 25 * x1**2 + 49 * x3**2 + 25 * x5**2 + 49 * x7**2 + 25 * y1**2 + 49 * y3**2 + 25 * y5**2 + 49 * y7**2 + 8

    variables = [x1, y1, x3, y3, x5, y5, x7, y7]
    equations = []
    for variable in variables:
        equations.append(2 * A * G * phi.derivative(variable) - 2 * phi * G * A.derivative(variable) - phi * A * G.derivative(variable))
    equations.append(1820**2 * W * A**2 * G - phi**2)
    return ring, equations, tuple(roots[n] for n in (2, 5, 7, 13))


def build_equivariant_modular_system_ordered(prime: int, order: str, return_data: bool = False, root_signs=None):
    field = GF(prime)
    roots = modular_roots(field, (2, 5, 7, 13), root_signs)
    rt2 = roots[2]
    rt5 = roots[5]
    rt7 = roots[7]
    rt13 = roots[13]

    def rt(n):
        factors = {
            26: rt2 * rt13,
            65: rt5 * rt13,
            91: rt7 * rt13,
            130: rt2 * rt5 * rt13,
            182: rt2 * rt7 * rt13,
            455: rt5 * rt7 * rt13,
            910: rt2 * rt5 * rt7 * rt13,
        }
        return factors[n]

    ring = PolynomialRing(
        field,
        ("x1", "y1", "x3", "y3", "x5", "y5", "x7", "y7", "W"),
        order=order,
    )
    x1, y1, x3, y3, x5, y5, x7, y7, W = ring.gens()

    def mul(z1, z2):
        return z1[0] * z2[0] - z1[1] * z2[1], z1[0] * z2[1] + z1[1] * z2[0]

    def conj(z):
        return z[0], -z[1]

    def scale(coef, z):
        real, imag = coef
        return real * z[0] - imag * z[1]

    u = (x1, y1)
    v = (x3, y3)
    ww = (x5, y5)
    rr = (x7, y7)

    phi = ring(0)
    phi += scale((-33 * rt(910) + field(525) * rt(26) / 2, 56 * rt(65) + 90 * rt(91)), mul(u, v))
    phi += scale((-field(525) * rt(26) / 2 - 33 * rt(910), 90 * rt(91) - 56 * rt(65)), mul(u, conj(v)))
    phi += scale((-field(21) * rt(130) / 2 + 90 * rt(182), -27 * rt(455) - 70 * rt13), mul(mul(u, v), conj(ww)))
    phi += scale((field(21) * rt(130) / 2 + 90 * rt(182), -70 * rt13 + 27 * rt(455)), mul(mul(conj(u), v), ww))
    phi += scale((23 * rt(455) + 189 * rt13, -69 * rt(182) + field(63) * rt(130) / 2), mul(mul(v, ww), rr))
    phi += scale((-23 * rt(455) + 189 * rt13, -69 * rt(182) - field(63) * rt(130) / 2), mul(mul(v, conj(ww)), rr))

    A = 5 * x1**2 + 7 * x3**2 + 5 * x5**2 + 7 * x7**2 + 5 * y1**2 + 7 * y3**2 + 5 * y5**2 + 7 * y7**2 + 2
    G = 25 * x1**2 + 49 * x3**2 + 25 * x5**2 + 49 * x7**2 + 25 * y1**2 + 49 * y3**2 + 25 * y5**2 + 49 * y7**2 + 8
    variables = [x1, y1, x3, y3, x5, y5, x7, y7]
    equations = [
        2 * A * G * phi.derivative(variable) - 2 * phi * G * A.derivative(variable) - phi * A * G.derivative(variable)
        for variable in variables
    ]
    equations.append(1820**2 * W * A**2 * G - phi**2)
    if return_data:
        return ring, equations, tuple(roots[n] for n in (2, 5, 7, 13)), A, G, phi
    return ring, equations, tuple(roots[n] for n in (2, 5, 7, 13))


def summarize_system(ring, equations) -> None:
    print(f"ring: {ring}")
    print(f"variables: {len(ring.gens())}")
    print(f"equations: {len(equations)}")
    for index, equation in enumerate(equations, 1):
        print(
            f"  E{index:02d}: total_degree={equation.degree():2d}, "
            f"terms={len(equation.monomials()):4d}"
        )


def modular_eliminate(prime: int, seconds: int, equivariant: bool = False) -> None:
    print(f"Building modular k3 elimination system over GF({prime})")
    if equivariant:
        ring, equations, roots = build_equivariant_modular_system(prime)
        print(f"sqrt choices modulo {prime}: sqrt2={roots[0]}, sqrt5={roots[1]}, sqrt7={roots[2]}, sqrt13={roots[3]}")
    else:
        ring, equations, roots = build_modular_system(prime)
        print(f"sqrt choices modulo {prime}: sqrt2={roots[0]}, sqrt5={roots[1]}, sqrt7={roots[2]}")
    summarize_system(ring, equations)

    W = ring.gens()[-1]
    ideal = ring.ideal(equations)
    eliminate_vars = list(ring.gens()[:-1])

    print(f"\nAttempting full elimination to GF({prime})[W] with timeout {seconds}s ...")
    start = cputime()
    try:
        alarm(seconds)
        elimination_ideal = ideal.elimination_ideal(eliminate_vars)
        cancel_alarm()
    except AlarmInterrupt:
        print(f"TIMEOUT after {seconds}s")
        return
    elapsed = cputime(start)
    print(f"elimination completed in {elapsed:.2f}s")

    generators = elimination_ideal.gens()
    print(f"elimination generators: {len(generators)}")
    for index, generator in enumerate(generators, 1):
        if generator == 0:
            continue
        print(f"  G{index}: degree={generator.degree(W)}, terms={len(generator.monomials())}")
        print(f"    {generator}")


def modular_groebner_probe(prime: int, seconds: int) -> None:
    print(f"Building grevlex modular k3 system over GF({prime})")
    ring, equations, roots = build_equivariant_modular_system_ordered(prime, "degrevlex")
    print(f"sqrt choices modulo {prime}: sqrt2={roots[0]}, sqrt5={roots[1]}, sqrt7={roots[2]}, sqrt13={roots[3]}")
    summarize_system(ring, equations)
    ideal = ring.ideal(equations)
    print(f"\nAttempting grevlex Groebner basis with timeout {seconds}s ...")
    start = cputime()
    try:
        alarm(seconds)
        basis = ideal.groebner_basis(algorithm="singular:std")
        cancel_alarm()
    except AlarmInterrupt:
        print(f"TIMEOUT after {seconds}s")
        return
    elapsed = cputime(start)
    print(f"grevlex basis completed in {elapsed:.2f}s")
    print(f"basis length: {len(basis)}")
    for index, generator in enumerate(basis[:20], 1):
        print(f"  G{index:02d}: degree={generator.degree():2d}, terms={len(generator.monomials()):5d}, lm={generator.lm()}")
    if len(basis) > 20:
        print(f"  ... {len(basis) - 20} more basis elements")


def modular_fglm_probe(prime: int, seconds: int) -> None:
    print(f"Building grevlex modular k3 system over GF({prime}) for FGLM")
    ring, equations, roots = build_equivariant_modular_system_ordered(prime, "degrevlex")
    print(f"sqrt choices modulo {prime}: sqrt2={roots[0]}, sqrt5={roots[1]}, sqrt7={roots[2]}, sqrt13={roots[3]}")
    summarize_system(ring, equations)
    ideal = ring.ideal(equations)

    start = cputime()
    try:
        alarm(seconds)
        grevlex_basis = ideal.groebner_basis(algorithm="singular:std")
        gb_ideal = ring.ideal(grevlex_basis)
        lex_ring = PolynomialRing(ring.base_ring(), ring.variable_names(), order="lex")
        lex_basis = gb_ideal.transformed_basis("fglm", other_ring=lex_ring)
        cancel_alarm()
    except AlarmInterrupt:
        print(f"TIMEOUT after {seconds}s")
        return
    elapsed = cputime(start)
    print(f"grevlex+FGLM completed in {elapsed:.2f}s")
    print(f"grevlex basis length: {len(grevlex_basis)}")
    print(f"lex basis length: {len(lex_basis)}")

    W = lex_ring.gens()[-1]
    w_only = []
    other_variables = set(lex_ring.gens()[:-1])
    for generator in lex_basis:
        variables = set(generator.variables())
        if variables and variables.issubset({W}):
            w_only.append(generator)
    print(f"univariate W generators: {len(w_only)}")
    for index, generator in enumerate(w_only, 1):
        print(f"  W{index}: degree={generator.degree(W)}, terms={len(generator.monomials())}")
        print(f"    {generator}")

    if not w_only:
        print("first 20 lex basis elements:")
        for index, generator in enumerate(lex_basis[:20], 1):
            print(f"  L{index:02d}: degree={generator.degree():2d}, terms={len(generator.monomials()):5d}, lm={generator.lm()}")


def modular_w_power_relation_probe(
    prime: int,
    seconds: int,
    max_power: int,
    progress_every: int,
    output: str | None,
    saturation: str,
    root_signs,
    gb_algorithm: str,
) -> None:
    """Find a W-only relation by reducing powers of W modulo a grevlex basis."""
    print(f"Building grevlex modular k3 system over GF({prime}) for W-power relation")
    ring, equations, roots, A, G, phi = build_equivariant_modular_system_ordered(
        prime,
        "degrevlex",
        return_data=True,
        root_signs=root_signs,
    )
    print(f"sqrt choices modulo {prime}: sqrt2={roots[0]}, sqrt5={roots[1]}, sqrt7={roots[2]}, sqrt13={roots[3]}")
    summarize_system(ring, equations)
    ideal = ring.ideal(equations)
    W = ring.gens()[-1]

    start = cputime()
    try:
        alarm(seconds)
        if saturation != "none":
            factor = A * G * phi
            if saturation == "target":
                for variable in ring.gens()[:-1]:
                    factor *= variable
            print(f"saturating by {saturation} factor ...")
            ideal, saturation_exponent = ideal.saturation(factor)
            print(f"saturation exponent: {saturation_exponent}")
        basis = ideal.groebner_basis(algorithm=f"singular:{gb_algorithm}")
        gb_ideal = ring.ideal(basis)
        exponent_index = {}
        vectors = []
        relation = None
        relation_degree = None
        reduced_power = gb_ideal.reduce(ring(1))
        for power in range(max_power + 1):
            if power == 0:
                reduced = reduced_power
            else:
                reduced_power = gb_ideal.reduce(reduced_power * W)
                reduced = reduced_power
            for exponent, coefficient in reduced.dict().items():
                if exponent not in exponent_index:
                    exponent_index[exponent] = len(exponent_index)
                    for old_vector in vectors:
                        old_vector.append(ring.base_ring()(0))
            coords = [ring.base_ring()(0)] * len(exponent_index)
            for exponent, coefficient in reduced.dict().items():
                coords[exponent_index[exponent]] = coefficient
            vectors.append(coords)
            if progress_every and power % progress_every == 0:
                print(f"  power={power:4d} vectors={len(vectors):4d} observed_coords={len(exponent_index):5d}")
            if len(vectors) <= len(exponent_index):
                continue
            matrix = Matrix(ring.base_ring(), [vector(ring.base_ring(), item) for item in vectors]).transpose()
            kernel = matrix.right_kernel()
            if kernel.dimension() > 0:
                raw = list(kernel.basis()[0])
                last = max(index for index, coefficient in enumerate(raw) if coefficient != 0)
                lead = raw[last]
                relation = [coefficient / lead for coefficient in raw[: last + 1]]
                relation_degree = last
                break
        cancel_alarm()
    except AlarmInterrupt:
        print(f"TIMEOUT after {seconds}s")
        return

    elapsed = cputime(start)
    print(f"grevlex basis and W-power reductions completed in {elapsed:.2f}s")
    print(f"basis length: {len(basis)}")
    print(f"observed reduced-coordinate dimension: {len(exponent_index)}")
    if relation is None:
        print(f"no W-power relation found through degree {max_power}")
        return

    print(f"W-power relation degree: {relation_degree}")
    polynomial = sum(ring.base_ring()(relation[index]) * W**index for index in range(relation_degree + 1))
    print(f"terms: {len(polynomial.monomials())}")
    print(polynomial)
    if output:
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "prime": int(prime),
                    "degree": int(relation_degree),
                    "terms": int(len(polynomial.monomials())),
                    "coefficients_low_to_high": [int(coefficient) for coefficient in relation],
                    "polynomial": str(polynomial),
                    "basis_length": int(len(basis)),
                    "observed_coordinate_dimension": int(len(exponent_index)),
                    "saturation": saturation,
                    "root_signs": {str(key): int(value) for key, value in root_signs.items()},
                    "sqrt_choices": {
                        "2": int(roots[0]),
                        "5": int(roots[1]),
                        "7": int(roots[2]),
                        "13": int(roots[3]),
                    },
                    "gb_algorithm": gb_algorithm,
                    "elapsed_seconds": float(elapsed),
                },
                handle,
                indent=2,
            )
        print(f"saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prime", type=int, default=0, help="prime modulus; choose automatically if omitted")
    parser.add_argument("--seconds", type=int, default=600, help="timeout for the elimination step")
    parser.add_argument("--summary-only", action="store_true", help="only build and summarize the modular system")
    parser.add_argument("--equivariant", action="store_true", help="use the smaller 8-variable polynomial equivariant system")
    parser.add_argument("--gb-probe", action="store_true", help="compute a grevlex Groebner basis for the smaller equivariant system")
    parser.add_argument("--fglm-probe", action="store_true", help="compute grevlex basis then FGLM-convert to lex for W elimination")
    parser.add_argument("--w-power-probe", action="store_true", help="reduce powers of W modulo the grevlex basis to find a W-only relation")
    parser.add_argument("--max-power", type=int, default=600, help="maximum W power for --w-power-probe")
    parser.add_argument("--progress-every", type=int, default=25, help="progress interval for --w-power-probe")
    parser.add_argument("--relation-output", default=None, help="optional JSON output path for --w-power-probe")
    parser.add_argument(
        "--saturation",
        choices=("none", "value", "target"),
        default="none",
        help="optional saturation for --w-power-probe; value uses A*G*Phi, target also multiplies all coordinates",
    )
    parser.add_argument("--root-signs", default=None, help="signs for sqrt(2),sqrt(5),sqrt(7),sqrt(13), e.g. '+-++'")
    parser.add_argument("--gb-algorithm", default="std", choices=("std", "slimgb"), help="Singular Groebner algorithm")
    args = parser.parse_args()

    prime = args.prime or quadratic_residue_prime(radicands=(2, 5, 7, 13) if args.equivariant else (2, 5, 7))
    if args.gb_probe:
        if not args.prime:
            prime = quadratic_residue_prime(radicands=(2, 5, 7, 13))
        modular_groebner_probe(prime, args.seconds)
        return
    if args.fglm_probe:
        if not args.prime:
            prime = quadratic_residue_prime(radicands=(2, 5, 7, 13))
        modular_fglm_probe(prime, args.seconds)
        return
    if args.w_power_probe:
        if not args.prime:
            prime = quadratic_residue_prime(radicands=(2, 5, 7, 13))
        root_signs = parse_root_signs(args.root_signs)
        modular_w_power_relation_probe(
            prime,
            args.seconds,
            args.max_power,
            args.progress_every,
            args.relation_output,
            args.saturation,
            root_signs,
            args.gb_algorithm,
        )
        return
    if args.equivariant:
        ring, equations, roots = build_equivariant_modular_system(prime)
    else:
        ring, equations, roots = build_modular_system(prime)
    if args.summary_only:
        if args.equivariant:
            print(f"Modular system over GF({prime}); sqrt2={roots[0]}, sqrt5={roots[1]}, sqrt7={roots[2]}, sqrt13={roots[3]}")
        else:
            print(f"Modular system over GF({prime}); sqrt2={roots[0]}, sqrt5={roots[1]}, sqrt7={roots[2]}")
        summarize_system(ring, equations)
        return

    modular_eliminate(prime, args.seconds, equivariant=args.equivariant)


if __name__ == "__main__":
    main()