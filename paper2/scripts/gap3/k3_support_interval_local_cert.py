#!/usr/bin/env python3
"""Interval local KKT certificate for replayed k=3 support mechanisms.

This is the rigorous-local counterpart of ``k3_shell_cluster_kkt_suite.py``.
It loads one replayable support row, fixes the symmetry/gauge flat directions by
choosing a positive-definite coordinate slice, and verifies on a small box:

* Krawczyk uniqueness for grad(-R)=0 in the slice;
* positive definiteness of the interval Hessian of -R in the slice;
* a value upper bound for R at the slice critical point below C3.

The interval Hessian is produced by second-order forward automatic
differentiation over mpmath intervals.  This is intended as the template for
turning the finite KKT ledger into interval/rational proof bricks.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np
from scipy.linalg import qr
from scipy.optimize import least_squares, minimize

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.gap3.k3_shell_cluster_kkt_suite import build_support_problem  # noqa: E402
from scripts.gap3.k3_shell_strata_scan import C3_TARGET, objective  # noqa: E402


C3_TARGET_TEXT = "0.021936469459403747249299192478957700397867315103825"


def interval_lower(value: mp.iv.mpf) -> float:
    return float(value.a)


def interval_upper(value: mp.iv.mpf) -> float:
    return float(value.b)


def interval_mid(value: mp.iv.mpf) -> float:
    return 0.5 * (interval_lower(value) + interval_upper(value))


def interval_abs_upper(value: mp.iv.mpf) -> float:
    return max(abs(interval_lower(value)), abs(interval_upper(value)))


def point_interval(value: float | str) -> mp.iv.mpf:
    text = str(value) if isinstance(value, str) else repr(float(value))
    return mp.iv.mpf([text, text])


@dataclass
class Jet:
    value: mp.iv.mpf
    grad: list[mp.iv.mpf]
    hess: list[list[mp.iv.mpf]]

    @staticmethod
    def constant(value: float | str | mp.iv.mpf, dimension: int) -> "Jet":
        if isinstance(value, mp.iv.mpf):
            interval = value
        else:
            interval = point_interval(value)
        zero = mp.iv.mpf([0, 0])
        return Jet(interval, [zero for _ in range(dimension)], [[zero for _ in range(dimension)] for _ in range(dimension)])

    @staticmethod
    def variable(value: mp.iv.mpf, dimension: int, index: int) -> "Jet":
        item = Jet.constant(value, dimension)
        item.grad[index] = mp.iv.mpf([1, 1])
        return item

    def __add__(self, other: "Jet" | float) -> "Jet":
        other = as_jet(other, len(self.grad))
        return Jet(
            self.value + other.value,
            [left + right for left, right in zip(self.grad, other.grad)],
            [[self.hess[i][j] + other.hess[i][j] for j in range(len(self.grad))] for i in range(len(self.grad))],
        )

    __radd__ = __add__

    def __neg__(self) -> "Jet":
        return Jet(-self.value, [-item for item in self.grad], [[-item for item in row] for row in self.hess])

    def __sub__(self, other: "Jet" | float) -> "Jet":
        return self + (-as_jet(other, len(self.grad)))

    def __rsub__(self, other: "Jet" | float) -> "Jet":
        return as_jet(other, len(self.grad)) - self

    def __mul__(self, other: "Jet" | float) -> "Jet":
        other = as_jet(other, len(self.grad))
        dimension = len(self.grad)
        grad = [self.grad[i] * other.value + self.value * other.grad[i] for i in range(dimension)]
        hess = []
        for i in range(dimension):
            row = []
            for j in range(dimension):
                row.append(
                    self.hess[i][j] * other.value
                    + self.grad[i] * other.grad[j]
                    + self.grad[j] * other.grad[i]
                    + self.value * other.hess[i][j]
                )
            hess.append(row)
        return Jet(self.value * other.value, grad, hess)

    __rmul__ = __mul__

    def unary(self, value: mp.iv.mpf, first: mp.iv.mpf, second: mp.iv.mpf) -> "Jet":
        dimension = len(self.grad)
        grad = [first * self.grad[i] for i in range(dimension)]
        hess = []
        for i in range(dimension):
            row = []
            for j in range(dimension):
                row.append(second * self.grad[i] * self.grad[j] + first * self.hess[i][j])
            hess.append(row)
        return Jet(value, grad, hess)

    def exp(self) -> "Jet":
        value = mp.iv.exp(self.value)
        return self.unary(value, value, value)

    def sin(self) -> "Jet":
        value = mp.iv.sin(self.value)
        return self.unary(value, mp.iv.cos(self.value), -value)

    def cos(self) -> "Jet":
        value = mp.iv.cos(self.value)
        return self.unary(value, -mp.iv.sin(self.value), -value)

    def sqrt(self) -> "Jet":
        value = mp.iv.sqrt(self.value)
        first = point_interval("0.5") / value
        second = -point_interval("0.25") / (self.value * value)
        return self.unary(value, first, second)

    def reciprocal(self) -> "Jet":
        value = 1 / self.value
        first = -value * value
        second = 2 * value * value * value
        return self.unary(value, first, second)

    def __truediv__(self, other: "Jet" | float) -> "Jet":
        return self * as_jet(other, len(self.grad)).reciprocal()

    def __rtruediv__(self, other: "Jet" | float) -> "Jet":
        return as_jet(other, len(self.grad)) / self


def as_jet(value: Jet | float, dimension: int) -> Jet:
    if isinstance(value, Jet):
        return value
    return Jet.constant(value, dimension)


@dataclass
class CJet:
    real: Jet
    imag: Jet

    @staticmethod
    def zero(dimension: int) -> "CJet":
        return CJet(Jet.constant(0.0, dimension), Jet.constant(0.0, dimension))

    def __add__(self, other: "CJet") -> "CJet":
        return CJet(self.real + other.real, self.imag + other.imag)


    def __sub__(self, other: "CJet") -> "CJet":
        return CJet(self.real - other.real, self.imag - other.imag)

    def __mul__(self, other: "CJet" | Jet | float) -> "CJet":
        if isinstance(other, CJet):
            return CJet(self.real * other.real - self.imag * other.imag, self.real * other.imag + self.imag * other.real)
        other = as_jet(other, len(self.real.grad))
        return CJet(self.real * other, self.imag * other)

    __rmul__ = __mul__

    def conj(self) -> "CJet":
        return CJet(self.real, -self.imag)


def cexp_i(angle: Jet) -> CJet:
    return CJet(angle.cos(), angle.sin())


def build_jets(params: np.ndarray, free_indices: list[int], radius: float, point: bool) -> list[Jet]:
    dimension = len(free_indices)
    free_position = {index: position for position, index in enumerate(free_indices)}
    jets = []
    for index, value in enumerate(params):
        if index in free_position:
            if point:
                interval = point_interval(float(value))
            else:
                interval = mp.iv.mpf([repr(float(value - radius)), repr(float(value + radius))])
            jets.append(Jet.variable(interval, dimension, free_position[index]))
        else:
            jets.append(Jet.constant(float(value), dimension))
    return jets


def support_objective_jet(problem: dict[str, Any], params: list[Jet]) -> Jet:
    dimension = len(params[0].grad) if params else 0
    u_pos: list[list[CJet]] = []
    x2 = Jet.constant(0.0, dimension)
    d2 = Jet.constant(0.0, dimension)
    for mode_index in range(int(problem["N"])):
        theta = params[4 * mode_index]
        phi = params[4 * mode_index + 1]
        psi = params[4 * mode_index + 2]
        loga = params[4 * mode_index + 3]
        amp = loga.exp()
        radius = (loga * 0.5).exp()
        ctheta = theta.cos()
        stheta = theta.sin()
        ephi = cexp_i(phi)
        epsi = cexp_i(psi)
        vector = []
        for coord in range(3):
            component = (ephi * (ctheta * float(problem["e1s"][mode_index, coord]))) + (epsi * (stheta * float(problem["e2s"][mode_index, coord])))
            vector.append(component * radius)
        u_pos.append(vector)
        shell = float(problem["k2s"][mode_index])
        x2 = x2 + amp * (2.0 * shell)
        d2 = d2 + amp * (2.0 * shell * shell)
    u_raw = u_pos + [[component.conj() for component in vector] for vector in u_pos]
    b_value = Jet.constant(0.0, dimension)
    for triad_index in range(len(problem["ell_idx"])):
        ell = int(problem["ell_idx"][triad_index])
        r_idx = int(problem["r_idx"][triad_index])
        s_idx = int(problem["s_idx"][triad_index])
        sdu = CJet.zero(dimension)
        ced = CJet.zero(dimension)
        for coord in range(3):
            sdu = sdu + u_raw[r_idx][coord] * float(problem["s_mat"][triad_index, coord])
            ced = ced + u_raw[ell][coord].conj() * u_raw[s_idx][coord]
        term = sdu * ced * float(problem["ell2"][triad_index])
        b_value = b_value - term.imag
    ratio = b_value / (x2 * d2.sqrt())
    return -ratio


def choose_free_indices(problem: dict[str, Any], params: np.ndarray, flat_tol: float) -> tuple[list[int], dict[str, Any]]:
    _, gradient = objective(problem, params)
    dimension = len(params)
    hessian = np.zeros((dimension, dimension), dtype=float)
    step = 1e-5
    for index in range(dimension):
        plus = params.copy()
        minus = params.copy()
        plus[index] += step
        minus[index] -= step
        _, grad_plus = objective(problem, plus)
        _, grad_minus = objective(problem, minus)
        hessian[:, index] = (np.asarray(grad_plus) - np.asarray(grad_minus)) / (2.0 * step)
    hessian = 0.5 * (hessian + hessian.T)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    positive = np.where(eigenvalues > flat_tol)[0]
    if len(positive) == 0:
        raise ValueError("no positive Hessian directions found")
    rank = int(len(positive))
    combination_count = math.comb(dimension, rank)
    slice_selection = "qr_pivot"
    if combination_count <= 20000:
        best_key = None
        best_indices = None
        for combo in itertools.combinations(range(dimension), rank):
            sliced_candidate = hessian[np.ix_(combo, combo)]
            sliced_eigenvalues = np.linalg.eigvalsh(sliced_candidate)
            sign, logdet = np.linalg.slogdet(sliced_candidate)
            key = (float(sliced_eigenvalues[0]), float(logdet) if sign > 0 else -math.inf)
            if best_key is None or key > best_key:
                best_key = key
                best_indices = combo
        if best_indices is None:
            raise RuntimeError("slice search failed")
        free_indices = [int(index) for index in best_indices]
        slice_selection = "exhaustive_max_min_eigen_then_logdet"
    else:
        q_matrix = eigenvectors[:, positive].T
        _, _, pivots = qr(q_matrix, pivoting=True)
        free_indices = sorted(int(index) for index in pivots[:rank])
    sliced = hessian[np.ix_(free_indices, free_indices)]
    sliced_eigenvalues = np.linalg.eigvalsh(sliced)
    return free_indices, {
        "full_dimension": dimension,
        "gradient_max_abs_float": float(np.max(np.abs(gradient))),
        "full_hessian_eigen_min": float(eigenvalues[0]),
        "full_hessian_eigen_max": float(eigenvalues[-1]),
        "positive_rank": rank,
        "slice_selection": slice_selection,
        "slice_combination_count": int(combination_count),
        "free_indices": free_indices,
        "slice_min_eigen_float": float(sliced_eigenvalues[0]),
        "slice_max_eigen_float": float(sliced_eigenvalues[-1]),
    }


def midpoint_matrix(hessian: list[list[mp.iv.mpf]]) -> np.ndarray:
    return np.asarray([[interval_mid(entry) for entry in row] for row in hessian], dtype=float)


def radius_matrix(hessian: list[list[mp.iv.mpf]]) -> np.ndarray:
    return np.asarray([[0.5 * (interval_upper(entry) - interval_lower(entry)) for entry in row] for row in hessian], dtype=float)


def interval_matrix_left_multiply(matrix: np.ndarray, vector: list[mp.iv.mpf]) -> list[mp.iv.mpf]:
    output = []
    for row in range(matrix.shape[0]):
        total = mp.iv.mpf([0, 0])
        for col in range(matrix.shape[1]):
            total += point_interval(float(matrix[row, col])) * vector[col]
        output.append(total)
    return output


def interval_matrix_product(matrix: np.ndarray, interval_matrix: list[list[mp.iv.mpf]]) -> list[list[mp.iv.mpf]]:
    rows = []
    for i in range(matrix.shape[0]):
        row = []
        for j in range(len(interval_matrix[0])):
            total = mp.iv.mpf([0, 0])
            for k in range(matrix.shape[1]):
                total += point_interval(float(matrix[i, k])) * interval_matrix[k][j]
            row.append(total)
        rows.append(row)
    return rows


def interval_matvec(matrix: list[list[mp.iv.mpf]], vector: list[mp.iv.mpf]) -> list[mp.iv.mpf]:
    rows = []
    for row in matrix:
        total = mp.iv.mpf([0, 0])
        for entry, component in zip(row, vector):
            total += entry * component
        rows.append(total)
    return rows


def krawczyk_check(gradient: list[mp.iv.mpf], hessian: list[list[mp.iv.mpf]], radius: float) -> dict[str, Any]:
    mid = midpoint_matrix(hessian)
    inverse_mid = np.linalg.inv(mid)
    y_gradient = interval_matrix_left_multiply(inverse_mid, gradient)
    y_hessian = interval_matrix_product(inverse_mid, hessian)
    identity_minus_yh = []
    dimension = len(gradient)
    for i in range(dimension):
        row = []
        for j in range(dimension):
            identity = mp.iv.mpf([1, 1]) if i == j else mp.iv.mpf([0, 0])
            row.append(identity - y_hessian[i][j])
        identity_minus_yh.append(row)
    delta = [mp.iv.mpf([repr(-radius), repr(radius)]) for _ in range(dimension)]
    remainder = interval_matvec(identity_minus_yh, delta)
    relative = [(-yg) + rem for yg, rem in zip(y_gradient, remainder)]
    subset = [interval_lower(item) > -radius and interval_upper(item) < radius for item in relative]
    return {
        "gradient_abs_upper": [interval_abs_upper(item) for item in gradient],
        "krawczyk_relative": [[interval_lower(item), interval_upper(item)] for item in relative],
        "subset_flags": subset,
        "passes": bool(all(subset)),
    }


def hessian_positive_check(hessian: list[list[mp.iv.mpf]]) -> dict[str, Any]:
    mid = 0.5 * (midpoint_matrix(hessian) + midpoint_matrix(hessian).T)
    rad = radius_matrix(hessian)
    rad = np.maximum(rad, rad.T)
    eig_min = float(np.linalg.eigvalsh(mid)[0])
    frob = float(np.linalg.norm(rad, ord="fro"))
    row_sum = float(np.max(np.sum(rad, axis=1)))
    radius_bound = min(frob, row_sum)
    lower = eig_min - radius_bound
    return {
        "lambda_min_midpoint": eig_min,
        "radius_bound": radius_bound,
        "frobenius_radius": frob,
        "row_sum_radius": row_sum,
        "certified_lower_bound": lower,
        "passes": bool(lower > 0.0),
    }


def load_cluster_row(path: str, cluster_index: int) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for row in data["rows"]:
        if int(row["cluster_index"]) == cluster_index:
            return row
    raise ValueError(f"cluster {cluster_index} not found in {path}")


def polish_slice(problem: dict[str, Any], params: np.ndarray, free_indices: list[int], maxiter: int) -> tuple[np.ndarray, dict[str, Any]]:
    start = params[free_indices].copy()

    def merge(free_values: np.ndarray) -> np.ndarray:
        merged = params.copy()
        merged[free_indices] = free_values
        return merged

    def fun(free_values: np.ndarray) -> float:
        value, _ = objective(problem, merge(free_values))
        return float(value)

    def jac(free_values: np.ndarray) -> np.ndarray:
        _, gradient = objective(problem, merge(free_values))
        return np.asarray(gradient, dtype=float)[free_indices]

    result = minimize(fun, start, jac=jac, method="BFGS", options={"gtol": 1e-13, "maxiter": maxiter})
    best_free = np.asarray(result.x, dtype=float)
    best_grad = jac(best_free)
    root_result = None
    if float(np.max(np.abs(best_grad))) > 1e-11:
        root_result = least_squares(
            jac,
            best_free,
            xtol=1e-14,
            ftol=1e-14,
            gtol=1e-14,
            max_nfev=maxiter * 20,
        )
        root_grad = jac(np.asarray(root_result.x, dtype=float))
        if float(np.max(np.abs(root_grad))) < float(np.max(np.abs(best_grad))):
            best_free = np.asarray(root_result.x, dtype=float)
            best_grad = root_grad
    polished = merge(best_free)
    value, gradient = objective(problem, polished)
    return polished, {
        "enabled": True,
        "success": bool(result.success or (root_result is not None and root_result.success)),
        "message": str(result.message),
        "root_message": None if root_result is None else str(root_result.message),
        "iterations": int(result.nit),
        "root_nfev": None if root_result is None else int(root_result.nfev),
        "value": float(-value),
        "free_gradient_max_abs": float(np.max(np.abs(np.asarray(gradient)[free_indices]))),
        "full_gradient_max_abs": float(np.max(np.abs(gradient))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kkt-json", required=True)
    parser.add_argument("--cluster", type=int, default=0)
    parser.add_argument("--radius", type=float, default=1e-7)
    parser.add_argument("--dps", type=int, default=60)
    parser.add_argument("--flat-tol", type=float, default=1e-7)
    parser.add_argument("--slice-polish", action="store_true")
    parser.add_argument("--slice-polish-maxiter", type=int, default=2000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    mp.iv.dps = args.dps
    row = load_cluster_row(args.kkt_json, args.cluster)
    support_modes = [tuple(int(component) for component in mode) for mode in row["support_modes"]]
    problem = build_support_problem(support_modes)
    params = np.asarray(row["support_params_normalized_x2_1"], dtype=float)
    free_indices, slice_info = choose_free_indices(problem, params, args.flat_tol)
    slice_polish = {"enabled": False}
    if args.slice_polish:
        params, slice_polish = polish_slice(problem, params, free_indices, args.slice_polish_maxiter)
        free_indices, slice_info = choose_free_indices(problem, params, args.flat_tol)

    point_jets = build_jets(params, free_indices, args.radius, point=True)
    point_objective = support_objective_jet(problem, point_jets)
    box_jets = build_jets(params, free_indices, args.radius, point=False)
    box_objective = support_objective_jet(problem, box_jets)
    krawczyk = krawczyk_check(point_objective.grad, box_objective.hess, args.radius)
    positivity = hessian_positive_check(box_objective.hess)

    gradient_norm = math.sqrt(sum(interval_abs_upper(item) ** 2 for item in point_objective.grad))
    value_correction = float("inf")
    if positivity["certified_lower_bound"] > 0:
        value_correction = 0.5 * gradient_norm * gradient_norm / positivity["certified_lower_bound"]
    r_center_upper = -interval_lower(point_objective.value)
    r_root_upper = r_center_upper + value_correction
    target_gap_lower = float(mp.mpf(C3_TARGET_TEXT) - mp.mpf(repr(r_root_upper)))

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": args.kkt_json,
        "cluster_index": args.cluster,
        "support_modes": [list(mode) for mode in support_modes],
        "radius": args.radius,
        "dps": args.dps,
        "slice": slice_info,
        "slice_polish": slice_polish,
        "center_minus_R_interval": [interval_lower(point_objective.value), interval_upper(point_objective.value)],
        "box_minus_R_interval": [interval_lower(box_objective.value), interval_upper(box_objective.value)],
        "krawczyk": krawczyk,
        "hessian_positive": positivity,
        "gradient_norm_abs_upper": gradient_norm,
        "value_correction_upper": value_correction,
        "R_critical_upper": r_root_upper,
        "target_gap_lower_using_text_target": target_gap_lower,
        "passes": bool(krawczyk["passes"] and positivity["passes"] and target_gap_lower > 0.0),
        "method": "interval AD2 Hessian/Krawczyk on a QR-selected gauge slice of the support-local KKT point",
    }
    print("k=3 support interval local certificate")
    print("=======================================")
    print(f"cluster={args.cluster} support={len(support_modes)} slice_dim={len(free_indices)} radius={args.radius:g}")
    print(f"Krawczyk={krawczyk['passes']} Hessian+={positivity['passes']} lower={positivity['certified_lower_bound']:+.6e}")
    print(f"R_critical_upper={r_root_upper:.17g} target_gap_lower={target_gap_lower:+.6e} passes={report['passes']}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()