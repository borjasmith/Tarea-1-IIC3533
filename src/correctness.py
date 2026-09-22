"""Validación de correctitud y reproducibilidad del inciso (c)."""

import argparse
import csv
from pathlib import Path

import numpy as np

from bs_auto import bootstrap_auto_v4
from bs_numpy import bootstrap_numpy_v2
from bs_sklearn import bootstrap_sklearn_v3
from data_generation import generate_data


METHODS = {
    "BaggingRegressor": lambda X, y, p: bootstrap_auto_v4(
        X, y, B=48, p=p, random_state=1234
    ),
    "sklearn + joblib": lambda X, y, p: bootstrap_sklearn_v3(
        X, y, B=48, p=p, base_seed=1234
    ),
    "NumPy + joblib": lambda X, y, p: bootstrap_numpy_v2(
        X, y, B=48, p=p, base_seed=1234
    ),
}


def interval_metrics(beta_star, lower, upper):
    return {
        "coverage_pct": 100.0
        * float(np.mean((beta_star >= lower) & (beta_star <= upper))),
        "mean_width": float(np.mean(upper - lower)),
    }


def max_interval_difference(first, second):
    return max(
        float(np.max(np.abs(first[0] - second[0]))),
        float(np.max(np.abs(first[1] - second[1]))),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/computer_1/correctness_c.csv")
    args = parser.parse_args()

    X, y, beta_star = generate_data(seed=1111, N=100_000, k=300)
    beta_hat = np.linalg.solve(X.T @ X, X.T @ y)
    relative_error = np.linalg.norm(beta_hat - beta_star) / np.linalg.norm(beta_star)
    print(f"error_relativo_beta_hat={relative_error:.12f}")

    rows = []
    reference_intervals = {}
    for name, method in METHODS.items():
        first = method(X, y, p=1)[:2]
        repeated = method(X, y, p=1)[:2]
        parallel = method(X, y, p=4)[:2]
        reference_intervals[name] = first

        metrics = interval_metrics(beta_star, *first)
        row = {
            "implementation": name,
            **metrics,
            "max_diff_repeat_p1": max_interval_difference(first, repeated),
            "max_diff_p1_vs_p4": max_interval_difference(first, parallel),
        }
        rows.append(row)
        print(
            f"{name}: cobertura={row['coverage_pct']:.3f}%, "
            f"ancho_medio={row['mean_width']:.6f}, "
            f"dif_repeticion={row['max_diff_repeat_p1']:.3e}, "
            f"dif_p1_p4={row['max_diff_p1_vs_p4']:.3e}",
            flush=True,
        )

    manual_difference = max_interval_difference(
        reference_intervals["sklearn + joblib"],
        reference_intervals["NumPy + joblib"],
    )
    print(f"max_diff_sklearn_vs_numpy={manual_difference:.3e}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
