"""Experimento del inciso (i): barrido de combinaciones (p, t) con p*t <= p_max.

Usa la versión más eficiente del inciso (b) (`bootstrap_numpy_v2`) y fija el
número de threads internos con `threadpool_limits` dentro de cada worker.

=============================================================================
INSTRUCCIONES PARA CORRERLO EN EL MAC
=============================================================================

¿Hay que correr algo antes?  NO.
    Este script no depende de ningún otro inciso: genera X e y internamente
    (misma semilla 1111, N=10000, k=300 que el resto de la tarea) y no lee
    ningún CSV de (f), (g) ni (h). Solo necesita el entorno de Python listo.

1. Entorno (una sola vez). Si el Mac no tiene el entorno conda de la tarea:

       conda create -n tarea1-hpc -c conda-forge --override-channels \
           python=3.13 numpy scikit-learn joblib threadpoolctl matplotlib -y
       conda activate tarea1-hpc

2. Verificar que threadpoolctl ve la librería BLAS (esto decide si el
   parámetro t tiene efecto real):

       python -c "from threadpoolctl import threadpool_info; print(threadpool_info())"

   - Sale una entrada 'openblas'  -> t es real; cada worker limita sus threads.
   - Sale una lista vacía []      -> NumPy usa Apple Accelerate (típico si se
     instaló con pip). threadpool_limits NO tiene efecto: para un mismo p los
     distintos t darán tiempos equivalentes salvo ruido y el script imprimirá
     "workers reportan [()]". Para que t sea controlable:

         conda install -c conda-forge "libblas=*=*openblas" numpy

     Si se decide correr igual con Accelerate (trade-off aceptado), anotarlo
     para decirlo explícitamente en el informe al comentar (i).

3. Correr SIEMPRE desde la raíz del repositorio (no desde src/):

       cd <raíz del repo>
       python src/experiment_i.py

   Por defecto escribe en la carpeta mac/ (se crea si no existe):
       mac/grid_i.csv     todas las mediciones (p, t, repetición, tiempo)
       mac/metrics_i.csv  mediana por combinación (p, t)
       mac/tiempos_i.png  ranking ordenado + mapa de calor t x p
   Con --output-dir se puede cambiar la carpeta; con --plot-only se redibuja la
   figura desde mac/grid_i.csv sin volver a medir.

4. Condiciones de medición: equipo enchufado, sin otras aplicaciones abiertas
   y sin usarlo durante la corrida (~2-3 min). No cambiar --p-max: debe ser el
   número de cores lógicos (8). Se usan B=48 y 3 repeticiones por combinación,
   igual que en el inciso (f).

5. Qué debe verse en pantalla al empezar (con OpenBLAS):
       verificación p=1, t=8: workers reportan [(8,)]
       verificación p=2, t=4: workers reportan [(4,)]
       verificación p=8, t=1: workers reportan [(1,)]
   y al final la lista de medianas con la mejor combinación marcada.

6. Después: los archivos de mac/ se mueven al informe a mano. Hoy el .tex
   espera la figura en results/computer_1/tiempos_i.png (secciones (i) y A.6);
   basta copiarla ahí o ajustar esa ruta. El cuadro de (i) se construye con la
   matriz t x p de metrics_i.csv, con el mismo formato del cuadro del Lenovo
   en el anexo A.6.
"""

import argparse
import csv
import os
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    # worker y la verificación no cubriría todos los procesos.
    time.sleep(0.15)
    return {
        "pid": os.getpid(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "threadpools": [
            (pool["internal_api"], pool["num_threads"]) for pool in threadpool_info()
        ],
    }


def limited_worker_info(threads):
    with threadpool_limits(limits=threads, user_api="blas"):
        return worker_info()


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
    parser.add_argument("--output-dir", default="mac")
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
