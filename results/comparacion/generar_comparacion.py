"""Figuras y tablas comparativas entre computer_1 y computer_2 para el inciso (j)."""
import csv, statistics as st
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "comparacion"
C = {"c1": ROOT / "computer_1", "c2": ROOT / "computer_2"}
LABEL = {"c1": "Computador 1 (M1, 8 cores, 8 GB)", "c2": "Computador 2 (M5, 10 cores, 24 GB)"}
SHORT = {"c1": "Comp. 1 (M1)", "c2": "Comp. 2 (M5)"}
COLOR = {"c1": "#2a78d6", "c2": "#eb6834"}
IMPLS = ["BaggingRegressor", "sklearn + joblib", "NumPy + joblib"]

plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12.5,
                     "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11})

def load_metrics(d):
    rows = list(csv.DictReader((d / "metrics_g.csv").open()))
    m = defaultdict(dict)
    for r in rows:
        m[r["implementation"]][int(r["p"])] = {k: float(r[k]) for k in ("time_s", "speedup", "efficiency")}
    return m

def load_overhead(d):
    o = defaultdict(dict)
    for r in csv.DictReader((d / "overhead_h.csv").open()):
        o[r["implementation"]][int(r["p"])] = float(r["overhead_s"])
    return o

def load_i(d):
    v = defaultdict(list)
    for r in csv.DictReader((d / "timings_i.csv").open()):
        v[(int(r["p"]), int(r["t"]))].append(float(r["time_s"]))
    return {k: st.median(x) for k, x in v.items()}

M = {k: load_metrics(v) for k, v in C.items()}
O = {k: load_overhead(v) for k, v in C.items()}
I = {k: load_i(v) for k, v in C.items()}
PMAX = {k: max(M[k][IMPLS[0]]) for k in C}

def panel_series(ax, key, metric, ideal=None):
    for c in ("c1", "c2"):
        ps = sorted(M[c][key]); ys = [M[c][key][p][metric] for p in ps]
        ax.plot(ps, ys, "o-", color=COLOR[c], lw=2, ms=6, label=SHORT[c])
    if ideal is not None:
        ps = list(range(1, max(PMAX.values()) + 1))
        ax.plot(ps, [ideal(p) for p in ps], "--", color="#7a8594", lw=1.5, label="Ideal")
    ax.set_xticks(range(1, max(PMAX.values()) + 1)); ax.grid(alpha=0.3); ax.set_xlabel("Número de procesos p")

# --- Figura 1: tiempos
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
for ax, key in zip(axes, IMPLS):
    panel_series(ax, key, "time_s"); ax.set_title(key); ax.set_ylabel("T(p) [s]"); ax.set_ylim(bottom=0)
axes[0].legend()
fig.suptitle("Tiempo de ejecución T(p) en ambos computadores (N = 100 000, B = 48)", fontsize=15)
fig.tight_layout(); fig.savefig(OUT / "comparacion_tiempos.png", dpi=200); plt.close(fig)

# --- Figura 2: speedup y eficiencia
fig, axes = plt.subplots(2, 3, figsize=(16, 8.6))
for j, key in enumerate(IMPLS):
    panel_series(axes[0, j], key, "speedup", ideal=lambda p: p)
    axes[0, j].set_title(key); axes[0, j].set_ylabel("S(p)"); axes[0, j].set_ylim(0, 4.2)
    axes[0, j].text(0.98, 0.95, "ideal S = p sale del gráfico en p = 4", transform=axes[0, j].transAxes,
                    ha="right", va="top", fontsize=9.5, color="#7a8594")
    panel_series(axes[1, j], key, "efficiency", ideal=lambda p: 1.0)
    axes[1, j].set_ylabel("E(p)"); axes[1, j].set_ylim(0, 1.08)
axes[0, 0].legend(loc="upper left")
fig.suptitle("Speedup S(p) y eficiencia E(p) en ambos computadores", fontsize=15)
fig.tight_layout(); fig.savefig(OUT / "comparacion_speedup_eficiencia.png", dpi=200); plt.close(fig)

# --- Figura 3: overhead
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
for ax, key in zip(axes, IMPLS):
    for c in ("c1", "c2"):
        ps = sorted(O[c][key]); ax.plot(ps, [O[c][key][p] for p in ps], "o-", color=COLOR[c], lw=2, ms=6, label=SHORT[c])
    ax.axhline(0, color="black", lw=1, ls="--"); ax.set_title(key); ax.set_ylabel(r"$T_o(p)=pT(p)-T(1)$ [s]")
    ax.set_xticks(range(1, max(PMAX.values()) + 1)); ax.grid(alpha=0.3); ax.set_xlabel("Número de procesos p")
axes[0].legend()
fig.suptitle("Overhead total en ambos computadores", fontsize=15)
fig.tight_layout(); fig.savefig(OUT / "comparacion_overhead.png", dpi=200); plt.close(fig)

# --- Figura 4: mapas (p,t)
fig, axes = plt.subplots(1, 2, figsize=(16, 6.4))
for ax, c in zip(axes, ("c1", "c2")):
    pm = PMAX[c]; vals = I[c]
    lo, hi = min(vals.values()), max(vals.values())
    import numpy as np
    grid = np.full((pm, pm), np.nan)
    for (p, t), v in vals.items(): grid[p - 1, t - 1] = v
    im = ax.imshow(grid, cmap="Blues", vmin=lo, vmax=hi, aspect="auto")
    best = min(vals, key=vals.get)
    for (p, t), v in vals.items():
        ax.text(t - 1, p - 1, f"{v:.2f}", ha="center", va="center", fontsize=9.5,
                color="white" if (v - lo) / (hi - lo + 1e-9) > 0.55 else "black",
                fontweight="bold" if (p, t) == best else "normal")
    ax.add_patch(plt.Rectangle((best[1] - 1.5, best[0] - 1.5), 1, 1, fill=False, ec="#eb6834", lw=2.5))
    ax.set_xticks(range(pm)); ax.set_xticklabels(range(1, pm + 1)); ax.set_yticks(range(pm)); ax.set_yticklabels(range(1, pm + 1))
    ax.set_xlabel("Threads solicitados t"); ax.set_ylabel("Procesos p")
    ax.set_title(f"{LABEL[c]}\nmejor: p={best[0]}, t={best[1]}, {vals[best]:.3f} s", fontsize=12.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("Tiempo mediano [s]")
fig.suptitle(r"Tiempo mediano para combinaciones $(p,t)$ con $p\,t \leq p_{\max}$, versión NumPy", fontsize=15)
fig.tight_layout(); fig.savefig(OUT / "comparacion_i.png", dpi=200); plt.close(fig)

# --- Tablas LaTeX
def f(x, n=3): return f"{x:.{n}f}"
L = []
L.append("% Tablas comparativas para el inciso (j). Generadas por results/comparacion/generar_comparacion.py\n")
L.append(r"""\begin{table}[h]
    \centering
    \small
    \begin{tabular}{|l|l|l|}
    \hline
    \textbf{Característica} & \textbf{Computador 1} & \textbf{Computador 2} \\
    \hline
    Chip & Apple M1 & Apple M5 \\
    Cores de rendimiento / eficiencia & 4 / 4 & 4 / 6 \\
    Cores lógicos ($p_{\max}$) & 8 & 10 \\
    Memoria RAM & 8 GB & 24 GB \\
    Sistema operativo & macOS 14.7.8 (arm64) & macOS 26.6.2 (arm64) \\
    Python / NumPy / scikit-learn / joblib & 3.9.6 / 2.0.2 / 1.6.1 / 1.5.3 & 3.9.6 / 2.0.2 / 1.6.1 / 1.5.3 \\
    Backend BLAS & Apple Accelerate & Apple Accelerate \\
    Repeticiones por configuración & 3 & 5 \\
    Combinaciones $(p,t)$ en (i) & 20 & 27 \\
    \hline
    \end{tabular}
    \caption{Características de los dos computadores. El software es idéntico; solo cambia el hardware.}
    \label{tab:hardware-j}
\end{table}
""")
# T(p) side by side
hdr = " & ".join([f"\\multicolumn{{2}}{{c|}}{{\\textbf{{{k}}}}}" for k in IMPLS])
rows = []
for p in range(1, 11):
    cells = []
    for k in IMPLS:
        for c in ("c1", "c2"):
            cells.append(f(M[c][k][p]["time_s"]) if p in M[c][k] else "--")
    rows.append(f"    {p} & " + " & ".join(cells) + r" \\")
L.append(r"""\begin{table}[h]
    \centering
    \scriptsize
    \begin{tabular}{|c|rr|rr|rr|}
    \hline
     & """ + hdr + r""" \\
    $p$ & C1 & C2 & C1 & C2 & C1 & C2 \\
    \hline
""" + "\n".join(rows) + r"""
    \hline
    \end{tabular}
    \caption{Tiempo mediano $T(p)$ en segundos en ambos computadores (C1: M1, C2: M5). El computador 1 llega hasta $p=8$.}
    \label{tab:tiempos-j}
\end{table}
""")
# Resumen S, E, overhead
rows = []
for k in IMPLS:
    cells = [k]
    for c in ("c1", "c2"):
        best = max(M[c][k], key=lambda p: M[c][k][p]["speedup"])
        pm = PMAX[c]
        cells += [f(M[c][k][1]["time_s"], 2), f"{M[c][k][best]['speedup']:.3f} ($p={best}$)", f(M[c][k][best]["efficiency"]),
                  f(M[c][k][8]["speedup"]), f(O[c][k][8], 1)]
    rows.append("    " + " & ".join(cells) + r" \\")
L.append(r"""\begin{table}[h]
    \centering
    \scriptsize
    \begin{tabular}{|l|rrrrr|rrrrr|}
    \hline
     & \multicolumn{5}{c|}{\textbf{Computador 1 (M1)}} & \multicolumn{5}{c|}{\textbf{Computador 2 (M5)}} \\
    \textbf{Implementación} & $T(1)$ & $S_{\max}$ & $E$ en $S_{\max}$ & $S(8)$ & $T_o(8)$ & $T(1)$ & $S_{\max}$ & $E$ en $S_{\max}$ & $S(8)$ & $T_o(8)$ \\
    \hline
""" + "\n".join(rows) + r"""
    \hline
    \end{tabular}
    \caption{Resumen de escalabilidad por computador. $T(1)$ y $T_o(8)$ en segundos; $S(8)$ permite comparar ambos equipos con el mismo número de procesos.}
    \label{tab:resumen-j}
\end{table}
""")
# razones C1/C2
rows = []
for k in IMPLS:
    cells = [k] + [f"{M['c1'][k][p]['time_s'] / M['c2'][k][p]['time_s']:.2f}" for p in (1, 2, 4, 8)]
    rows.append("    " + " & ".join(cells) + r" \\")
L.append(r"""\begin{table}[h]
    \centering
    \small
    \begin{tabular}{|l|r|r|r|r|}
    \hline
    \textbf{Implementación} & $p=1$ & $p=2$ & $p=4$ & $p=8$ \\
    \hline
""" + "\n".join(rows) + r"""
    \hline
    \end{tabular}
    \caption{Razón $T_1(p)/T_2(p)$: cuántas veces más rápido es el computador 2 que el computador 1 para el mismo $p$.}
    \label{tab:razon-j}
\end{table}
""")
# mejor (p,t)
rows = []
for c in ("c1", "c2"):
    best = min(I[c], key=I[c].get); one = [I[c][(1, t)] for t in range(1, PMAX[c] + 1)]
    rows.append(f"    {LABEL[c]} & $({best[0]},{best[1]})$ & {I[c][best]:.3f} & {I[c][(1,1)]:.3f} & {min(one):.3f}--{max(one):.3f} \\\\")
L.append(r"""\begin{table}[h]
    \centering
    \small
    \begin{tabular}{|l|c|r|r|r|}
    \hline
    \textbf{Computador} & \textbf{Mejor $(p,t)$} & \textbf{Tiempo [s]} & \textbf{$(1,1)$ [s]} & \textbf{Rango fila $p=1$ [s]} \\
    \hline
""" + "\n".join(rows) + r"""
    \hline
    \end{tabular}
    \caption{Mejor combinación del inciso (i) en cada computador. La última columna muestra cuánto varía el tiempo con $p=1$ al cambiar $t$; un rango estrecho indica que el límite $t$ no tuvo efecto.}
    \label{tab:pt-j}
\end{table}
""")
(OUT / "tablas_comparacion.tex").write_text("\n".join(L), encoding="utf-8")

# resumen numérico
print("Razones T1/T2 por p:")
for k in IMPLS:
    print(" ", k, {p: round(M['c1'][k][p]['time_s'] / M['c2'][k][p]['time_s'], 2) for p in range(1, 9)})
print("listo:", sorted(p.name for p in OUT.iterdir()))
