"""Analyze which shells carry the one-high annulus linear envelope."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


def load_shell_norms(path: Path, base_s2: int) -> tuple[float, float, float, list[tuple[int, float]]]:
    shell_l2_sq: dict[int, float] = {}
    x2 = d2 = base_ratio = None
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if int(row["fixed_base_s2"]) != base_s2:
                continue
            shell_value = int(row["shell"])
            shell_l2_sq[shell_value] = shell_l2_sq.get(shell_value, 0.0) + float(row["shell_l2"]) ** 2
            x2 = float(row["x2"])
            d2 = float(row["d2"])
            base_ratio = float(row["base_ratio"])
    if x2 is None or d2 is None or base_ratio is None:
        raise ValueError(f"no rows for base_s2={base_s2} in {path}")
    rows = [(shell_value, shell_l2_sq[shell_value] ** 0.5) for shell_value in sorted(shell_l2_sq)]
    return x2, d2, base_ratio, rows


def value_and_grad(
    t: np.ndarray,
    gradient_norms: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    base_ratio: float,
) -> tuple[float, np.ndarray]:
    t2 = t * t
    one_plus_alpha = 1.0 + float(alpha @ t2)
    one_plus_beta = 1.0 + float(beta @ t2)
    numerator = base_ratio + float(gradient_norms @ t)
    denominator = one_plus_alpha * np.sqrt(one_plus_beta)
    value = numerator / denominator
    dlog_denominator = (2.0 * alpha * t / one_plus_alpha) + (beta * t / one_plus_beta)
    grad = (gradient_norms - numerator * dlog_denominator) / denominator
    return value, grad


def optimize(rows: list[tuple[int, float]], x2: float, d2: float, base_ratio: float) -> dict[str, object]:
    shells = np.asarray([shell_value for shell_value, _ in rows], dtype=np.float64)
    gradient_norms = np.asarray([norm for _, norm in rows], dtype=np.float64)
    alpha = 2.0 * shells / x2
    beta = 2.0 * shells * shells / d2
    q_local = base_ratio * (alpha + 0.5 * beta)
    initial = gradient_norms / (2.0 * q_local)

    def objective(t: np.ndarray) -> tuple[float, np.ndarray]:
        value, grad = value_and_grad(t, gradient_norms, alpha, beta, base_ratio)
        return -value, -grad

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, None)] * len(rows),
        options={"ftol": 1e-15, "gtol": 1e-13, "maxiter": 2000, "maxls": 50},
    )
    t = np.asarray(result.x, dtype=np.float64)
    value, _ = value_and_grad(t, gradient_norms, alpha, beta, base_ratio)
    contribution = gradient_norms * t
    return {
        "success": bool(result.success),
        "iterations": int(result.nit),
        "shells": shells,
        "gradient_norms": gradient_norms,
        "t": t,
        "contribution": contribution,
        "value": float(value),
        "base_ratio": base_ratio,
        "alpha": alpha,
        "beta": beta,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path, default=Path("scripts/results/cstar_one_high_annulus_s565.csv"), nargs="?")
    parser.add_argument("--base-s2", type=int, default=565)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--top-k", type=int, nargs="*", default=[1, 3, 5, 10, 20, 30, 50, 100, 200])
    args = parser.parse_args()

    x2, d2, base_ratio, rows = load_shell_norms(args.csv_path, args.base_s2)
    result = optimize(rows, x2, d2, base_ratio)
    shells = result["shells"]
    gradient_norms = result["gradient_norms"]
    t = result["t"]
    contribution = result["contribution"]
    alpha = result["alpha"]
    beta = result["beta"]
    value = float(result["value"])
    order = np.argsort(contribution)[::-1]
    total_contribution = float(np.sum(contribution))
    total_gain = value - base_ratio

    print(f"success={result['success']} iterations={result['iterations']}")
    print(f"base_ratio={base_ratio:.17g}")
    print(f"exact_linear_ratio={value:.17g}")
    print(f"exact_linear_gain={total_gain:.17g}")
    print(f"shells={len(shells)} active_shells={int(np.count_nonzero(t > 1e-14))}")
    print(f"total_linear_numerator_contribution={total_contribution:.17g}")
    print("rank shell gradient_norm t linear_contribution cumulative_fraction delta_X2_fraction delta_D2_fraction")
    cumulative = 0.0
    for rank, index in enumerate(order[: args.top], start=1):
        cumulative += float(contribution[index])
        dx = float(alpha[index] * t[index] * t[index])
        dd = float(beta[index] * t[index] * t[index])
        print(
            f"{rank} {int(shells[index])} {gradient_norms[index]:.9e} {t[index]:.9e} "
            f"{contribution[index]:.9e} {cumulative / total_contribution:.9f} {dx:.9e} {dd:.9e}"
        )

    for count in args.top_k:
        indices = order[: min(count, len(order))]
        restricted_t = np.zeros_like(t)
        restricted_t[indices] = t[indices]
        restricted_value, _ = value_and_grad(restricted_t, gradient_norms, alpha, beta, base_ratio)
        restricted_gain = restricted_value - base_ratio
        print(
            f"topK={count} ratio={restricted_value:.17g} gain={restricted_gain:.9e} "
            f"gain_fraction={restricted_gain / total_gain:.9f} "
            f"shell_min={int(np.min(shells[indices]))} shell_max={int(np.max(shells[indices]))}"
        )


if __name__ == "__main__":
    main()