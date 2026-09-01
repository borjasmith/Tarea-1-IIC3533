"""Experimento del inciso (e): procesos, threads BLAS y oversubscription."""

import argparse
import os

from joblib import Parallel, delayed
from threadpoolctl import threadpool_info, threadpool_limits

from bs_numpy import bootstrap_numpy_v2
from data_generation import generate_data


def worker_info():
    """Devuelve el PID y los pools numéricos visibles en un worker de joblib."""
    return {"pid": os.getpid(), "threadpools": threadpool_info()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", "-p", type=int, required=True)
    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        default=None,
        help="Límite de threads internos aplicado mediante threadpoolctl",
    )
    args = parser.parse_args()

    X, y, _ = generate_data(seed=1111, N=10_000, k=300)

    # Se consulta dentro de los workers, no solamente en el proceso principal.
    workers = Parallel(n_jobs=args.processes)(
        delayed(worker_info)() for _ in range(max(args.processes * 2, 1))
    )
    unique_workers = {item["pid"]: item["threadpools"] for item in workers}

    with threadpool_limits(limits=args.threads):
        _, _, elapsed = bootstrap_numpy_v2(
            X, y, B=48, p=args.processes, base_seed=1234
        )

    print(f"cores_logicos={os.cpu_count()}")
    print(f"p={args.processes}, limite_threads={args.threads}")
    print(f"workers={unique_workers}")
    print(f"threadpool_principal={threadpool_info()}")
    print(f"tiempo_s={elapsed:.6f}")


if __name__ == "__main__":
    main()
