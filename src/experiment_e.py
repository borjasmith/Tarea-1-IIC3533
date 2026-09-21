"""Experimento del inciso (e): procesos, threads BLAS y oversubscription."""

import argparse
import csv
import json
import os
import time
from pathlib import Path

from joblib import Parallel, delayed
from threadpoolctl import threadpool_info, threadpool_limits

from bs_numpy import bootstrap_numpy_v2
from data_generation import generate_data


def worker_info():
    """Devuelve el PID y los pools numéricos visibles en un worker de joblib."""
    # Mantiene la tarea activa brevemente para que todos los workers participen.
    time.sleep(0.05)
    return {"pid": os.getpid(), "threadpools": threadpool_info()}


def run_configuration(X, y, processes, threads):
    workers = Parallel(n_jobs=processes)(
        delayed(worker_info)() for _ in range(max(processes * 4, 1))
    )
    unique_workers = {item["pid"]: item["threadpools"] for item in workers}

    with threadpool_limits(limits=threads):
        _, _, elapsed = bootstrap_numpy_v2(X, y, B=48, p=processes, base_seed=1234)

    return {
        "p": processes,
        "thread_limit": threads,
        "workers_observed": len(unique_workers),
        "worker_pids": json.dumps(sorted(unique_workers)),
        "worker_threadpools": json.dumps(unique_workers, sort_keys=True),
        "main_threadpools": json.dumps(threadpool_info(), sort_keys=True),
        "time_s": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", "-p", type=int, default=1)
    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        default=None,
        help="Límite de threads internos aplicado mediante threadpoolctl",
    )
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", default="results/computer_1/observations_e.csv")
    args = parser.parse_args()

    X, y, _ = generate_data(seed=1111, N=10_000, k=300)
    if args.suite:
        rows = []
        for p in (1, 2, 4, 8):
            for repetition in range(1, args.repetitions + 1):
                row = run_configuration(X, y, p, args.threads)
                row["repetition"] = repetition
                rows.append(row)
                print(
                    f"p={p}, repetición={repetition}, "
                    f"workers={row['workers_observed']}, tiempo={row['time_s']:.6f}s",
                    flush=True,
                )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as file:
            fields = ["p", "repetition", "thread_limit", "workers_observed",
                      "worker_pids", "worker_threadpools", "main_threadpools", "time_s"]
            writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    else:
        row = run_configuration(X, y, args.processes, args.threads)
        print(f"cores_logicos={os.cpu_count()}")
        for key, value in row.items():
            print(f"{key}={value}")


if __name__ == "__main__":
    main()
