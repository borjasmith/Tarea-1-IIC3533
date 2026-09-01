"""Benchmark reproducible del inciso (f).

Ejecuta las tres implementaciones finales para p=1,...,p_max, guarda cada
medición en CSV y produce un gráfico resumen usando la mediana.
"""

import argparse
import csv
import os
import platform
import random
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn

from bs_auto import bootstrap_auto_v4
from bs_numpy import bootstrap_numpy_v2
from bs_sklearn import bootstrap_sklearn_v3
from data_generation import generate_data


IMPLEMENTATIONS = {
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


def write_metadata(path, p_max, repetitions):
    path.write_text(
        "\n".join(
            [
                f"sistema={platform.platform()}",
                f"procesador={platform.processor() or platform.machine()}",
                f"cores_logicos={os.cpu_count()}",
                f"python={platform.python_version()}",
                f"numpy={np.__version__}",
                f"scikit_learn={sklearn.__version__}",
                f"joblib={joblib.__version__}",
                f"p_max={p_max}",
                f"repeticiones={repetitions}",
                "N=10000",
                "k=300",
                "B=48",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def save_plot(rows, path, p_max):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in IMPLEMENTATIONS:
        medians = []
        lower_errors = []
        upper_errors = []
        for p in range(1, p_max + 1):
            values = [
                row["time_s"]
                for row in rows
                if row["implementation"] == name and row["p"] == p
            ]
            median = float(np.median(values))
            medians.append(median)
            lower_errors.append(median - min(values))
            upper_errors.append(max(values) - median)
        ax.errorbar(
            range(1, p_max + 1),
            medians,
            yerr=[lower_errors, upper_errors],
            marker="o",
            capsize=3,
            linewidth=1.8,
            label=name,
        )

    ax.set_title("Tiempo de ejecución según número de procesos")
    ax.set_xlabel("Número de procesos p")
    ax.set_ylabel("Tiempo T(p) [s]")
    ax.set_xticks(range(1, p_max + 1))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p-max", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--repetitions", "-r", type=int, default=3)
    parser.add_argument("--output-dir", default="results/computer_1")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_metadata(output_dir / "metadata.txt", args.p_max, args.repetitions)

    X, y, _ = generate_data(seed=1111, N=10_000, k=300)
    configurations = [
        (name, p, repetition)
        for repetition in range(1, args.repetitions + 1)
        for p in range(1, args.p_max + 1)
        for name in IMPLEMENTATIONS
    ]
    # Evita favorecer sistemáticamente a una implementación por el orden.
    random.Random(1111).shuffle(configurations)

    rows = []
    total = len(configurations)
    for index, (name, p, repetition) in enumerate(configurations, start=1):
        _, _, elapsed = IMPLEMENTATIONS[name](X, y, p)
        row = {
            "implementation": name,
            "p": p,
            "repetition": repetition,
            "time_s": elapsed,
        }
        rows.append(row)
        print(
            f"[{index:02d}/{total}] {name}, p={p}, repetición={repetition}: "
            f"{elapsed:.6f} s",
            flush=True,
        )

    rows.sort(key=lambda row: (row["implementation"], row["p"], row["repetition"]))
    with (output_dir / "timings_f.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    save_plot(rows, output_dir / "tiempos_f.png", args.p_max)

    print("\nMedianas [s]")
    for name in IMPLEMENTATIONS:
        values = []
        for p in range(1, args.p_max + 1):
            samples = [
                row["time_s"]
                for row in rows
                if row["implementation"] == name and row["p"] == p
            ]
            values.append(float(np.median(samples)))
        print(f"{name}: " + ", ".join(f"{value:.6f}" for value in values))


if __name__ == "__main__":
    main()
