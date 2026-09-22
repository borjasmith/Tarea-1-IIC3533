"""Experimento (p,t) del inciso (i) usando la implementación NumPy."""

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from threadpoolctl import threadpool_info, threadpool_limits

from data_generation import N_OBS, generate_data


def fit_numpy_limited(X, y, seed_b, threads):
    with threadpool_limits(limits=threads):
        rng = np.random.default_rng(seed_b)
        indices = rng.choice(X.shape[0], size=X.shape[0], replace=True)
        X_b = X[indices]
        y_b = y[indices]
        return np.linalg.solve(X_b.T @ X_b, X_b.T @ y_b)


def bootstrap_numpy_limited(X, y, B, processes, threads, base_seed=1234):
    start = time.perf_counter()
    coefficients = Parallel(n_jobs=processes)(
        delayed(fit_numpy_limited)(X, y, base_seed + b, threads) for b in range(B)
    )
    coefficients = np.asarray(coefficients)
    np.percentile(coefficients, [2.5, 97.5], axis=0)
    return time.perf_counter() - start


def worker(processes, threads, n_obs):
    X, y, _ = generate_data(N=n_obs)
    elapsed = bootstrap_numpy_limited(X, y, 48, processes, threads)
    return {"time_s": elapsed, "visible_threadpools": threadpool_info()}


def save_plot(rows, path, p_max):
    fig, ax = plt.subplots(figsize=(8, 5))
    thread_values = sorted({row["t"] for row in rows})
    for threads in thread_values:
        selected = [row for row in rows if row["t"] == threads]
        medians = []
        processes = sorted({row["p"] for row in selected})
        for p in processes:
            values = [row["time_s"] for row in selected if row["p"] == p]
            medians.append(float(np.median(values)))
        ax.plot(processes, medians, "o-", label=f"t={threads}")
    ax.set_title(rf"Tiempo para combinaciones $(p,t)$ con $p\,t\leq {p_max}$")
    ax.set_xlabel("Número de procesos p")
    ax.set_ylabel("Tiempo mediano [s]")
    ax.set_xticks(range(1, p_max + 1))
    ax.grid(alpha=0.3)
    ax.legend(title="Límite solicitado t", ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p-max", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output-dir", default="results/computer_1")
    parser.add_argument("--worker-p", type=int)
    parser.add_argument("--worker-t", type=int)
    parser.add_argument("--n-obs", type=int, default=N_OBS)
    args = parser.parse_args()

    if args.worker_p is not None:
        if args.worker_t is None:
            parser.error("--worker-t es obligatorio en modo worker")
        print("RESULT_JSON=" + json.dumps(worker(args.worker_p, args.worker_t, args.n_obs)))
        return

    combinations = [
        (p, t, repetition)
        for p in range(1, args.p_max + 1)
        for t in range(1, args.p_max + 1)
        if p * t <= args.p_max
        for repetition in range(1, args.repetitions + 1)
    ]
    random.Random(3533).shuffle(combinations)
    rows = []
    total = len(combinations)
    for index, (p, t, repetition) in enumerate(combinations, start=1):
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-p",
                str(p),
                "--worker-t",
                str(t),
                "--n-obs",
                str(args.n_obs),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        result_line = next(
            line for line in completed.stdout.splitlines() if line.startswith("RESULT_JSON=")
        )
        result = json.loads(result_line.removeprefix("RESULT_JSON="))
        rows.append(
            {
                "p": p,
                "t": t,
                "p_times_t": p * t,
                "repetition": repetition,
                "time_s": result["time_s"],
                "visible_threadpools": json.dumps(result["visible_threadpools"]),
            }
        )
        print(
            f"[{index:03d}/{total}] p={p}, t={t}, repetición={repetition}: "
            f"{result['time_s']:.6f} s",
            flush=True,
        )

    rows.sort(key=lambda row: (row["p"], row["t"], row["repetition"]))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "timings_i.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    save_plot(rows, output_dir / "tiempos_i.png", args.p_max)

    medians = []
    for p, t in sorted({(row["p"], row["t"]) for row in rows}):
        values = [row["time_s"] for row in rows if row["p"] == p and row["t"] == t]
        medians.append((float(np.median(values)), p, t))
    for median, p, t in medians:
        print(f"mediana p={p}, t={t}: {median:.6f} s")
    best = min(medians)
    print(f"mejor combinación: p={best[1]}, t={best[2]}, mediana={best[0]:.6f} s")


if __name__ == "__main__":
    main()
