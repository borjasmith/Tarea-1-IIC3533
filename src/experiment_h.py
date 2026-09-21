"""Cálculo del overhead total T_o(p)=p*T(p)-T(1) para el inciso (h)."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "BaggingRegressor": "tab:blue",
    "sklearn + joblib": "tab:orange",
    "NumPy + joblib": "tab:green",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/computer_1/metrics_g.csv")
    parser.add_argument("--output-dir", default="results/computer_1")
    args = parser.parse_args()

    with Path(args.input).open(encoding="utf-8") as file:
        metrics = list(csv.DictReader(file))

    rows = []
    for row in metrics:
        name = row["implementation"]
        p = int(row["p"])
        time_s = float(row["time_s"])
        baseline = next(
            float(item["time_s"])
            for item in metrics
            if item["implementation"] == name and int(item["p"]) == 1
        )
        rows.append(
            {
                "implementation": name,
                "p": p,
                "time_s": time_s,
                "baseline_s": baseline,
                "overhead_s": p * time_s - baseline,
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "overhead_h.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, color in COLORS.items():
        selected = [row for row in rows if row["implementation"] == name]
        ax.plot(
            [row["p"] for row in selected],
            [row["overhead_s"] for row in selected],
            "o-",
            color=color,
            label=name,
        )
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_title("Overhead total en función del número de procesos")
    ax.set_xlabel("Número de procesos p")
    ax.set_ylabel(r"$T_o(p)=pT(p)-T(1)$ [s]")
    ax.set_xticks(range(1, 9))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "overhead_h.png", dpi=200)
    plt.close(fig)

    for name in COLORS:
        values = [row for row in rows if row["implementation"] == name]
        print(name + ": " + ", ".join(f"{row['overhead_s']:.3f}" for row in values))


if __name__ == "__main__":
    main()
