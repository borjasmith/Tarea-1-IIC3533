"""Tiempos por versión del inciso (b).

Cronometra las 10 versiones iterativas de las tres implementaciones con p=1,
repitiendo cada medición y reportando la mediana, para alimentar los cuadros
comparativos del informe.
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np

from bs_auto import (
    bootstrap_auto_v1,
    bootstrap_auto_v2,
    bootstrap_auto_v3,
    bootstrap_auto_v4,
    bootstrap_auto_v5,
)
from bs_numpy import bootstrap_numpy_v1, bootstrap_numpy_v2
from bs_sklearn import (
    bootstrap_sklearn_v1,
    bootstrap_sklearn_v2,
    bootstrap_sklearn_v3,
)
from data_generation import generate_data


# El orden replica el de los cuadros del informe.
VERSIONS = [
    ("bs_auto.py", "v1", lambda X, y, B: bootstrap_auto_v1(X, y, B=B, p=1)),
    ("bs_auto.py", "v2", lambda X, y, B: bootstrap_auto_v2(X, y, B=B, p=1)),
    ("bs_auto.py", "v3", lambda X, y, B: bootstrap_auto_v3(X, y, B=B, p=1)),
    ("bs_auto.py", "v4", lambda X, y, B: bootstrap_auto_v4(X, y, B=B, p=1, random_state=1234)),
    ("bs_auto.py", "v5", lambda X, y, B: bootstrap_auto_v5(X, y, B=B, p=1)),
    ("bs_sklearn.py", "v1", lambda X, y, B: bootstrap_sklearn_v1(X, y, B=B, p=1)),
    ("bs_sklearn.py", "v2", lambda X, y, B: bootstrap_sklearn_v2(X, y, B=B, p=1, base_seed=1234)),
    ("bs_sklearn.py", "v3", lambda X, y, B: bootstrap_sklearn_v3(X, y, B=B, p=1, base_seed=1234)),
    ("bs_numpy.py", "v1", lambda X, y, B: bootstrap_numpy_v1(X, y, B=B, p=1, base_seed=1234)),
    ("bs_numpy.py", "v2", lambda X, y, B: bootstrap_numpy_v2(X, y, B=B, p=1, base_seed=1234)),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", "-r", type=int, default=3)
    parser.add_argument("--resamples", "-B", type=int, default=48)
    parser.add_argument("--output-dir", default="results/computer_1")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, _ = generate_data(seed=1111, N=10_000, k=300)

    configurations = [
        (index, repetition)
        for repetition in range(1, args.repetitions + 1)
        for index in range(len(VERSIONS))
    ]
    # Misma precaución que en (f): el orden no debe favorecer a una versión.
    random.Random(1111).shuffle(configurations)

    rows = []
    total = len(configurations)
    for position, (index, repetition) in enumerate(configurations, start=1):
        script, version, run = VERSIONS[index]
        _, _, elapsed = run(X, y, args.resamples)
        rows.append(
            {
                "script": script,
                "version": version,
                "repetition": repetition,
                "time_s": elapsed,
            }
        )
        print(
            f"[{position:02d}/{total}] {script} {version}, "
            f"repetición={repetition}: {elapsed:.6f} s",
            flush=True,
        )

    rows.sort(key=lambda row: (row["script"], row["version"], row["repetition"]))
    csv_path = output_dir / "versions_b.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\nMedianas por versión [s]")
    for script, version, _ in VERSIONS:
        samples = [
            row["time_s"]
            for row in rows
            if row["script"] == script and row["version"] == version
        ]
        print(f"{script:<15} {version}: {float(np.median(samples)):.4f}")

    print(f"\nCSV generado: {csv_path}")


if __name__ == "__main__":
    main()
