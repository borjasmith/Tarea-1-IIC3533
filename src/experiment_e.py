"""Experimento del inciso (e): procesos, threads BLAS y oversubscription."""

import argparse
import os
import time

import numpy as np
from joblib import Parallel, delayed
from threadpoolctl import threadpool_info, threadpool_limits

from bs_numpy import fit_numpy_v2
from data_generation import generate_data


def worker_info():
    """PID, límite heredado y pools numéricos visibles dentro de un worker."""
    # Una operación de álgebra lineal fuerza la carga efectiva de BLAS.
    matrix = np.random.rand(200, 200)
    matrix @ matrix
    # Sin esta pausa las tareas son tan breves que loky las resuelve en un solo
    # worker y el conteo de procesos distintos queda subestimado.
    time.sleep(0.15)
    return {
        "pid": os.getpid(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "threadpools": [
            (pool["internal_api"], pool["num_threads"]) for pool in threadpool_info()
        ],
    }


def fit_limited(X, y, seed_b, threads):
    """Ajuste de un resample con el límite de threads aplicado en el worker.

    Envolver la llamada a `Parallel` en el proceso padre no sirve: loky arranca
    los workers con su propio entorno, así que `threadpool_limits` debe
    aplicarse dentro de la tarea para que BLAS lo respete.
    """
    with threadpool_limits(limits=threads, user_api="blas"):
        return fit_numpy_v2(X, y, seed_b)


def bootstrap_numpy_pt(X, y, B, p, t, base_seed=1234):
    """Igual que `bootstrap_numpy_v2`, pero fijando t threads por proceso."""
    start_time = time.time()
    seeds = [base_seed + i for i in range(B)]

    coefs = Parallel(n_jobs=p)(
        delayed(fit_limited)(X, y, seeds[i], t) for i in range(B)
    )

    coefs = np.array(coefs)
    lower_bound = np.percentile(coefs, 2.5, axis=0)
    upper_bound = np.percentile(coefs, 97.5, axis=0)

    return lower_bound, upper_bound, time.time() - start_time


def probe_workers(p, threads):
    """Consulta los pools dentro de los workers, no solo en el padre."""
    # threadpool_limits(limits=None) no impone nada: deja el reparto de joblib.
    reports = Parallel(n_jobs=p, batch_size=1)(
        delayed(limited_worker_info)(threads) for _ in range(max(p * 2, 1))
    )
    return {report["pid"]: report for report in reports}


def limited_worker_info(threads):
    with threadpool_limits(limits=threads, user_api="blas"):
        return worker_info()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", "-p", type=int, required=True)
    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        default=None,
        help="Límite de threads internos aplicado con threadpoolctl dentro de "
        "cada worker. Si se omite, se usa el reparto por defecto de joblib.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Veces que se repite el bootstrap completo. Con B=48 una sola "
        "pasada dura menos de un segundo, insuficiente para observar los "
        "procesos en el monitor del sistema; con --repeat 30 dura ~30 s.",
    )
    args = parser.parse_args()

    X, y, _ = generate_data(seed=1111, N=10_000, k=300)

    workers = probe_workers(args.processes, args.threads)

    if args.repeat > 1:
        print(
            f"Ejecutando {args.repeat} pasadas con p={args.processes}: "
            "observa ahora el monitor del sistema...",
            flush=True,
        )
    elapsed_runs = []
    for _ in range(args.repeat):
        _, _, elapsed = bootstrap_numpy_pt(
            X, y, B=48, p=args.processes, t=args.threads, base_seed=1234
        )
        elapsed_runs.append(elapsed)
    elapsed = float(np.median(elapsed_runs))

    print(f"cores_logicos={os.cpu_count()}")
    print(f"p={args.processes}, limite_threads={args.threads}")
    print(f"procesos_trabajadores_distintos={len(workers)}")
    for pid, report in workers.items():
        print(
            f"  worker pid={pid} OMP_NUM_THREADS={report['omp_num_threads']} "
            f"pools={report['threadpools']}"
        )
    print(
        "threadpool_principal="
        f"{[(pool['internal_api'], pool['num_threads']) for pool in threadpool_info()]}"
    )
    print(f"pasadas={args.repeat}")
    print(f"tiempo_s={elapsed:.6f}  (mediana de las pasadas)")


if __name__ == "__main__":
    main()
