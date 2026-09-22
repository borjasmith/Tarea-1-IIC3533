"""Medición de las versiones evolutivas del inciso (b)."""

import csv
from pathlib import Path

from bs_auto import (
    bootstrap_auto_v1,
    bootstrap_auto_v2,
    bootstrap_auto_v3,
    bootstrap_auto_v4,
    bootstrap_auto_v5,
)
from bs_numpy import bootstrap_numpy_v1, bootstrap_numpy_v2
from bs_sklearn import bootstrap_sklearn_v1, bootstrap_sklearn_v2, bootstrap_sklearn_v3
from data_generation import generate_data


METHODS = [
    ("auto", "v1", lambda X, y: bootstrap_auto_v1(X, y, B=48, p=1)),
    ("auto", "v2", lambda X, y: bootstrap_auto_v2(X, y, B=48, p=1)),
    ("auto", "v3", lambda X, y: bootstrap_auto_v3(X, y, B=48, p=1)),
    ("auto", "v4", lambda X, y: bootstrap_auto_v4(X, y, B=48, p=1)),
    ("auto", "v5", lambda X, y: bootstrap_auto_v5(X, y, B=48, p=1)),
    ("sklearn", "v1", lambda X, y: bootstrap_sklearn_v1(X, y, B=48, p=1)),
    ("sklearn", "v2", lambda X, y: bootstrap_sklearn_v2(X, y, B=48, p=1)),
    ("sklearn", "v3", lambda X, y: bootstrap_sklearn_v3(X, y, B=48, p=1)),
    ("numpy", "v1", lambda X, y: bootstrap_numpy_v1(X, y, B=48, p=1)),
    ("numpy", "v2", lambda X, y: bootstrap_numpy_v2(X, y, B=48, p=1)),
]


def main():
    X, y, _ = generate_data(seed=1111, N=100_000, k=300)
    output = Path("results/computer_1/timings_b.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["family", "version", "p", "time_s"], lineterminator="\n"
        )
        writer.writeheader()
        for family, version, method in METHODS:
            _, _, elapsed = method(X, y)
            writer.writerow({"family": family, "version": version, "p": 1, "time_s": elapsed})
            file.flush()
            print(f"{family} {version}: {elapsed:.6f} s", flush=True)


if __name__ == "__main__":
    main()
