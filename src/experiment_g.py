"""Cálculo y visualización de T(p), S(p) y E(p) para el inciso (g)."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "BaggingRegressor": "tab:blue",
    "sklearn + joblib": "tab:orange",
    "NumPy + joblib": "tab:green",
}


def load_medians(path):
    with path.open(encoding="utf-8") as file:
        raw_rows = list(csv.DictReader(file))

    implementations = list(COLORS)
    p_values = sorted({int(row["p"]) for row in raw_rows})
    metrics = []
    for name in implementations:
        times = {}
        for p in p_values:
            samples = [
                float(row["time_s"])
                for row in raw_rows
                if row["implementation"] == name and int(row["p"]) == p
            ]
            if not samples:
                raise ValueError(f"Faltan mediciones para {name}, p={p}")
            times[p] = float(np.median(samples))

        baseline = times[1]
        for p in p_values:
            speedup = baseline / times[p]
            metrics.append(
                {
                    "implementation": name,
                    "p": p,
                    "time_s": times[p],
                    "speedup": speedup,
                    "efficiency": speedup / p,
                    "ideal_time_s": baseline / p,
                    "ideal_speedup": float(p),
                    "ideal_efficiency": 1.0,
                }
            )
    return metrics, p_values


def write_metrics(metrics, path):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)


def plot_metrics(metrics, p_values, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    for name, color in COLORS.items():
        rows = [row for row in metrics if row["implementation"] == name]
        axes[0].plot(
            p_values, [row["time_s"] for row in rows], "o-", color=color, label=name
        )
        axes[0].plot(
            p_values,
            [row["ideal_time_s"] for row in rows],
            "--",
            color=color,
            alpha=0.55,
        )
        axes[1].plot(
            p_values, [row["speedup"] for row in rows], "o-", color=color, label=name
        )
        axes[2].plot(
            p_values,
            [row["efficiency"] for row in rows],
            "o-",
            color=color,
            label=name,
        )

    axes[1].plot(p_values, p_values, "k--", label="Ideal")
    axes[2].axhline(1.0, color="black", linestyle="--", label="Ideal")

    axes[0].set_title("Tiempo")
    axes[0].set_ylabel("T(p) [s]")
    axes[1].set_title("Speedup")
    axes[1].set_ylabel("S(p)")
    axes[2].set_title("Eficiencia")
    axes[2].set_ylabel("E(p)")
    for ax in axes:
        ax.set_xlabel("Número de procesos p")
        ax.set_xticks(p_values)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    axes[2].legend(fontsize=8)

    fig.suptitle("Escalabilidad de las implementaciones de bootstrap")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/computer_1/timings_f.csv")
    parser.add_argument("--output-dir", default="results/computer_1")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, p_values = load_medians(Path(args.input))
    write_metrics(metrics, output_dir / "metrics_g.csv")
    plot_metrics(metrics, p_values, output_dir / "metricas_g.png")

    for name in COLORS:
        rows = [row for row in metrics if row["implementation"] == name]
        best = max(rows, key=lambda row: row["speedup"])
        print(
            f"{name}: T(1)={rows[0]['time_s']:.6f} s; "
            f"máximo S={best['speedup']:.3f} y E={best['efficiency']:.3f} "
            f"en p={best['p']}"
        )


if __name__ == "__main__":
    main()
