"""Experimento del inciso (i): barrido de combinaciones (p, t) con p*t <= p_max.

Usa la versión más eficiente del inciso (b) (`bootstrap_numpy_v2`) y fija el
número de threads internos con `threadpool_limits` dentro de cada worker.
"""

import argparse
import csv
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from data_generation import generate_data
from experiment_e import bootstrap_numpy_pt, limited_worker_info


def build_grid(p_max):
    """Combinaciones (p, t) con p, t >= 1 y p*t <= p_max."""
    return [
        (p, t)
        for p in range(1, p_max + 1)
        for t in range(1, p_max + 1)
        if p * t <= p_max
    ]


def report_worker_threads(p, t):
    """Verifica que el límite de threads llegue efectivamente a los workers."""
    reports = Parallel(n_jobs=p)(delayed(limited_worker_info)(t) for _ in range(p))
    return sorted(
        {tuple(threads for _, threads in report["threadpools"]) for report in reports}
    )


def save_plot(rows, grid, path, p_max):
    t_values = sorted({t for _, t in grid})
    p_values = sorted({p for p, _ in grid})

    medians = {}
    for p, t in grid:
        samples = [
            row["time_s"] for row in rows if row["p"] == p and row["t"] == t
        ]
        medians[(p, t)] = float(np.median(samples))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))

    # Panel izquierdo: ranking de las combinaciones. Un gráfico de líneas no
    # sirve acá porque los t grandes solo admiten p=1 y quedan puntos sueltos.
    ranking = sorted(medians.items(), key=lambda item: item[1])
    labels = [f"p={p}, t={t}" for (p, t), _ in ranking]
    values = [value for _, value in ranking]
    colors = ["crimson" if index == 0 else "tab:blue" for index in range(len(values))]

    positions = np.arange(len(values))
    axes[0].barh(positions, values, color=colors)
    axes[0].set_yticks(positions, labels, fontsize=9.5)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Tiempo T(p, t) [s]")
    axes[0].set_title("Combinaciones ordenadas por tiempo mediano")
    axes[0].grid(axis="x", alpha=0.3)
    for position, value in zip(positions, values):
        axes[0].text(
            value, position, f" {value:.3f}", va="center", fontsize=8.5
        )
    axes[0].set_xlim(0, max(values) * 1.12)

    best = ranking[0][0]

    matrix = np.full((len(t_values), len(p_values)), np.nan)
    for row_index, t in enumerate(t_values):
        for column_index, p in enumerate(p_values):
            if (p, t) in medians:
                matrix[row_index, column_index] = medians[(p, t)]

    image = axes[1].imshow(matrix, cmap="viridis_r", origin="lower", aspect="auto")
    axes[1].set_xticks(range(len(p_values)), [str(p) for p in p_values])
    axes[1].set_yticks(range(len(t_values)), [str(t) for t in t_values])
    axes[1].set_xlabel("Número de procesos p")
    axes[1].set_ylabel("Threads internos t")
    axes[1].set_title(f"Mediana [s]; celdas vacías: $p\\,t > {p_max}$")
    for row_index in range(len(t_values)):
        for column_index in range(len(p_values)):
            value = matrix[row_index, column_index]
            if not np.isnan(value):
                axes[1].text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if value > np.nanmedian(matrix) else "black",
                )
    # Recuadro sobre la mejor combinación, para que coincida con el ranking.
    axes[1].add_patch(
        plt.Rectangle(
            (p_values.index(best[0]) - 0.5, t_values.index(best[1]) - 0.5),
            1,
            1,
            fill=False,
            edgecolor="crimson",
            linewidth=2.5,
        )
    )
    fig.colorbar(image, ax=axes[1], label="Tiempo [s]")

    fig.suptitle(
        "Inciso (i): procesos vs. threads internos con "
        f"$p\\,t \\leq {p_max}$ (NumPy + joblib)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

    return medians, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p-max", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--repetitions", "-r", type=int, default=3)
    parser.add_argument("--resamples", "-B", type=int, default=48)
    parser.add_argument("--output-dir", default="results/computer_1")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="No vuelve a medir: redibuja la figura a partir de grid_i.csv.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = build_grid(args.p_max)

    if args.plot_only:
        with (output_dir / "grid_i.csv").open(encoding="utf-8") as file:
            rows = [
                {"p": int(r["p"]), "t": int(r["t"]),
                 "repetition": int(r["repetition"]), "time_s": float(r["time_s"])}
                for r in csv.DictReader(file)
            ]
        _, best = save_plot(rows, grid, output_dir / "tiempos_i.png", args.p_max)
        print(f"Gráfico regenerado: {output_dir / 'tiempos_i.png'} (mejor: p={best[0]}, t={best[1]})")
        return

    X, y, _ = generate_data(seed=1111, N=10_000, k=300)

    print(f"cores_logicos={os.cpu_count()}, p_max={args.p_max}")
    print(f"combinaciones (p,t) evaluadas: {len(grid)}")
    for p, t in [(1, args.p_max), (2, max(args.p_max // 2, 1)), (args.p_max, 1)]:
        if (p, t) in grid:
            print(f"  verificación p={p}, t={t}: workers reportan {report_worker_threads(p, t)}")

    configurations = [
        (p, t, repetition)
        for repetition in range(1, args.repetitions + 1)
        for p, t in grid
    ]
    # Igual que en (f): el orden aleatorio evita sesgos por calentamiento.
    random.Random(1111).shuffle(configurations)

    rows = []
    total = len(configurations)
    for index, (p, t, repetition) in enumerate(configurations, start=1):
        _, _, elapsed = bootstrap_numpy_pt(
            X, y, B=args.resamples, p=p, t=t, base_seed=1234
        )
        rows.append({"p": p, "t": t, "repetition": repetition, "time_s": elapsed})
        print(
            f"[{index:02d}/{total}] p={p}, t={t}, repetición={repetition}: "
            f"{elapsed:.6f} s",
            flush=True,
        )

    rows.sort(key=lambda row: (row["p"], row["t"], row["repetition"]))
    csv_path = output_dir / "grid_i.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    plot_path = output_dir / "tiempos_i.png"
    medians, best = save_plot(rows, grid, plot_path, args.p_max)

    medians_path = output_dir / "metrics_i.csv"
    with medians_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["p", "t", "time_s"])
        writer.writeheader()
        for (p, t), value in sorted(medians.items()):
            writer.writerow({"p": p, "t": t, "time_s": value})

    print("\nMedianas [s]")
    for (p, t), value in sorted(medians.items()):
        marker = "  <-- mejor" if (p, t) == best else ""
        print(f"  p={p}, t={t}: {value:.6f}{marker}")

    print(f"\nCSV generado: {csv_path}")
    print(f"CSV generado: {medians_path}")
    print(f"Gráfico generado: {plot_path}")


if __name__ == "__main__":
    main()
