import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Mismos colores para comparar
COLORS = {
    "BaggingRegressor": "tab:blue",
    "sklearn + joblib": "tab:orange",
    "NumPy + joblib": "tab:green",
}


def load_results(path):

    rows = []

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required = {"implementation", "p", "time_s"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(
                f"El CSV debe contener las columnas: {sorted(required)}"
            )

        for row in reader:
            rows.append(
                {
                    "implementation": row["implementation"],
                    "p": int(row["p"]),
                    "time_s": float(row["time_s"]),
                }
            )

    return rows


def calculate_overhead(rows):

    # Función core calcula el overhead con T(1) de cada implementación

    implementations = sorted(
        {row["implementation"] for row in rows}
    )

    overhead_rows = []

    for implementation in implementations:
        data = [
            row for row in rows
            if row["implementation"] == implementation
        ]
        data.sort(key=lambda row: row["p"])

        t1_values = [row["time_s"] for row in data if row["p"] == 1]

        if len(t1_values) != 1:
            raise ValueError(
                f"No se encontró exactamente un T(1) para {implementation}."
            )

        t1 = t1_values[0]

        for row in data:
            p = row["p"]
            time_s = row["time_s"]

            overhead = p * time_s - t1

            overhead_rows.append(
                {
                    "implementation": implementation,
                    "p": p,
                    "time_s": time_s,
                    "t1_s": t1,
                    "overhead_s": overhead,
                }
            )

    return overhead_rows


def save_results(rows, path):

    fieldnames = [
        "implementation",
        "p",
        "time_s",
        "t1_s",
        "overhead_s",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_overhead(rows, path):
    implementations = sorted(
        {row["implementation"] for row in rows}
    )

    p_values = sorted({row["p"] for row in rows})

    fig, ax = plt.subplots(figsize=(9, 6))

    for implementation in implementations:
        data = [
            row for row in rows
            if row["implementation"] == implementation
        ]
        data.sort(key=lambda row: row["p"])

        p = np.array([row["p"] for row in data])
        overhead = np.array([row["overhead_s"] for row in data])

        ax.plot(
            p,
            overhead,
            marker="o",
            color=COLORS.get(implementation),
            label=implementation,
        )

    ax.axhline(0.0, color="black", linestyle="--", label="Ideal")

    ax.set_xlabel("Número de procesos p")
    ax.set_ylabel("Overhead $T_o(p)$ [s]")
    ax.set_title("Overhead de la paralelización")
    ax.set_xticks(p_values)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def print_summary(rows):

    print("\nResultados del inciso (h)")
    print("-" * 75)
    print(
        f"{'Implementación':<25}"
        f"{'p':>4}"
        f"{'T(p) [s]':>14}"
        f"{'To(p) [s]':>16}"
    )
    print("-" * 75)

    for row in rows:
        print(
            f"{row['implementation']:<25}"
            f"{row['p']:>4}"
            f"{row['time_s']:>14.6f}"
            f"{row['overhead_s']:>16.6f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/computer_1/metrics_g.csv")
    parser.add_argument("--output-dir", default="results/computer_1")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"No se encontró '{input_path}'. "
            "Ejecuta primero experiment_g.py o pasa --input."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "metrics_h.csv"
    plot_path = output_dir / "overhead_h.png"

    rows = load_results(input_path)
    overhead_rows = calculate_overhead(rows)

    save_results(overhead_rows, csv_path)
    print_summary(overhead_rows)
    plot_overhead(overhead_rows, plot_path)

    print(f"\nCSV generado: {csv_path}")
    print(f"Gráfico generado: {plot_path}")


if __name__ == "__main__":
    main()
