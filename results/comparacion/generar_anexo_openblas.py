"""Figuras y tablas del anexo OpenBLAS (computador 2): (e) threads visibles, (i) mapa (p,t), oversubscription."""
import csv, json, statistics as st
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OB = ROOT / "computer_2_openblas"; AC = ROOT / "computer_2"; OUT = OB
PMAX = 10
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13.5, "axes.labelsize": 12.5, "xtick.labelsize": 11, "ytick.labelsize": 11})

def med_i(d):
    v = defaultdict(list)
    for r in csv.DictReader((d / "timings_i.csv").open()):
        v[(int(r["p"]), int(r["t"]))].append(float(r["time_s"]))
    return {k: st.median(x) for k, x in v.items()}, v

I_ob, I_ob_raw = med_i(OB); I_ac, _ = med_i(AC)

# --- Figura 1: mapas (p,t) Accelerate vs OpenBLAS, mismo computador
fig, axes = plt.subplots(1, 2, figsize=(16, 6.4))
for ax, (title, vals) in zip(axes, [("NumPy con Accelerate (pip): límite $t$ sin efecto", I_ac), ("NumPy con OpenBLAS (conda): límite $t$ efectivo", I_ob)]):
    lo, hi = min(vals.values()), max(vals.values())
    grid = np.full((PMAX, PMAX), np.nan)
    for (p, t), v in vals.items(): grid[p - 1, t - 1] = v
    im = ax.imshow(grid, cmap="Blues", vmin=lo, vmax=hi, aspect="auto")
    best = min(vals, key=vals.get)
    for (p, t), v in vals.items():
        ax.text(t - 1, p - 1, f"{v:.2f}", ha="center", va="center", fontsize=9.5,
                color="white" if (v - lo) / (hi - lo + 1e-9) > 0.55 else "black", fontweight="bold" if (p, t) == best else "normal")
    ax.add_patch(plt.Rectangle((best[1] - 1.5, best[0] - 1.5), 1, 1, fill=False, ec="#eb6834", lw=2.5))
    ax.set_xticks(range(PMAX)); ax.set_xticklabels(range(1, PMAX + 1)); ax.set_yticks(range(PMAX)); ax.set_yticklabels(range(1, PMAX + 1))
    ax.set_xlabel("Threads solicitados t"); ax.set_ylabel("Procesos p")
    ax.set_title(f"{title}\nmejor: p={best[0]}, t={best[1]}, {vals[best]:.3f} s", fontsize=12.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("Tiempo mediano [s]")
fig.suptitle("Computador 2: mismo experimento (i) con dos backends BLAS", fontsize=15)
fig.tight_layout(); fig.savefig(OUT / "tiempos_i_openblas_vs_accelerate.png", dpi=200); plt.close(fig)

# --- Figura 2: fila p=1 en función de t (evidencia directa)
fig, ax = plt.subplots(figsize=(9, 4.6))
ts = list(range(1, PMAX + 1))
ax.plot(ts, [I_ac[(1, t)] for t in ts], "o-", color="#2a78d6", lw=2, ms=6, label="Accelerate (pip)")
ax.plot(ts, [I_ob[(1, t)] for t in ts], "o-", color="#eb6834", lw=2, ms=6, label="OpenBLAS (conda)")
ax.set_xticks(ts); ax.set_xlabel("Threads solicitados t (con p = 1)"); ax.set_ylabel("Tiempo mediano [s]"); ax.set_ylim(bottom=0)
ax.grid(alpha=0.3); ax.legend(); ax.set_title("Un solo proceso: ¿el límite de threads cambia el tiempo?")
fig.tight_layout(); fig.savefig(OUT / "fila_p1_openblas_vs_accelerate.png", dpi=200); plt.close(fig)

# --- Oversubscription
ov = defaultdict(list)
for r in csv.DictReader((OB / "timings_oversub.csv").open()):
    ov[(int(r["p"]), int(r["t"]))].append(float(r["time_s"]))
ov_med = {k: st.median(v) for k, v in ov.items()}
best_ob = min(I_ob, key=I_ob.get)
bars = [(f"({best_ob[0]},{best_ob[1]})\np·t={best_ob[0]*best_ob[1]}", I_ob[best_ob], "#1baf7a")]
bars += [(f"({p},{t})\np·t={p*t}", ov_med[(p, t)], "#e34948") for (p, t) in sorted(ov_med, key=lambda k: k[0] * k[1])]
fig, ax = plt.subplots(figsize=(10, 5))
xs = range(len(bars))
ax.bar(xs, [b[1] for b in bars], color=[b[2] for b in bars], width=0.65)
for i, b in enumerate(bars): ax.text(i, b[1] * 1.08, f"{b[1]:.1f} s", ha="center", fontsize=11)
ax.set_xticks(list(xs)); ax.set_xticklabels([b[0] for b in bars]); ax.set_yscale("log")
ax.set_ylabel("Tiempo mediano [s], escala log"); ax.set_title("Oversubscription real con OpenBLAS: combinaciones con p·t > 10 cores")
ax.grid(alpha=0.3, axis="y"); ax.set_ylim(top=max(b[1] for b in bars) * 2.2)
fig.tight_layout(); fig.savefig(OUT / "oversubscription_openblas.png", dpi=200); plt.close(fig)

# --- (e): observations
obs = list(csv.DictReader((OB / "observations_e.csv").open()))
e_rows = defaultdict(list); e_threads = {}
for r in obs:
    p = int(r["p"]); e_rows[p].append(float(r["time_s"]))
    pools = json.loads(r["worker_threadpools"])
    nts = sorted({pl["num_threads"] for lst in pools.values() for pl in lst if pl.get("internal_api") == "openblas"})
    e_threads[p] = (int(r["workers_observed"]), nts)

# --- Tablas LaTeX
L = ["% Tablas del anexo OpenBLAS (computador 2). Generadas por results/comparacion/generar_anexo_openblas.py\n"]
rows = "\n".join(f"    {p} & {e_threads[p][0]} & {', '.join(map(str, e_threads[p][1])) or '--'} & {p * (e_threads[p][1][0] if e_threads[p][1] else 0)} & {st.median(e_rows[p]):.3f} \\\\" for p in sorted(e_rows))
L.append(r"""\begin{table}[H]
    \centering
    \small
    \begin{tabular}{|c|c|c|c|c|}
    \hline
    $p$ & \textbf{Workers} & \textbf{Threads OpenBLAS por worker} & \textbf{Threads totales} & \textbf{Tiempo mediano [s]} \\
    \hline
""" + rows + r"""
    \hline
    \end{tabular}
    \caption{Anexo (e), computador 2 con OpenBLAS: \texttt{threadpool\_info()} dentro de cada worker sí reporta el pool y su número de threads. Sin límite, cada proceso abre 10 threads.}
    \label{tab:anexo-e-openblas}
\end{table}
""")
top = sorted(I_ob.items(), key=lambda kv: kv[1])[:5]
rows = "\n".join(f"    $({p},{t})$ & {p*t} & {m:.3f} & {min(I_ob_raw[(p,t)]):.3f} & {max(I_ob_raw[(p,t)]):.3f} \\\\" for (p, t), m in top)
L.append(r"""\begin{table}[H]
    \centering
    \small
    \begin{tabular}{|c|c|r|r|r|}
    \hline
    \textbf{$(p,t)$} & $p\,t$ & \textbf{Mediana [s]} & \textbf{Mín [s]} & \textbf{Máx [s]} \\
    \hline
""" + rows + r"""
    \hline
    \end{tabular}
    \caption{Anexo (i), computador 2 con OpenBLAS: cinco mejores combinaciones (5 repeticiones cada una).}
    \label{tab:anexo-i-top}
\end{table}
""")
rows = "\n".join(f"    $(1,{t})$ & {I_ac[(1,t)]:.3f} & {I_ob[(1,t)]:.3f} \\\\" for t in ts)
L.append(r"""\begin{table}[H]
    \centering
    \small
    \begin{tabular}{|c|r|r|}
    \hline
    \textbf{$(p,t)$} & \textbf{Accelerate [s]} & \textbf{OpenBLAS [s]} \\
    \hline
""" + rows + r"""
    \hline
    \end{tabular}
    \caption{Fila $p=1$ del inciso (i) en el computador 2 con ambos backends. Con Accelerate el tiempo no depende de $t$; con OpenBLAS sí.}
    \label{tab:anexo-i-p1}
\end{table}
""")
rows = f"    $({best_ob[0]},{best_ob[1]})$ & {best_ob[0]*best_ob[1]} & {I_ob[best_ob]:.3f} & 1.0 \\\\\n"
rows += "\n".join(f"    $({p},{t})$ & {p*t} & {ov_med[(p,t)]:.3f} & {ov_med[(p,t)]/I_ob[best_ob]:.1f} \\\\" for (p, t) in sorted(ov_med, key=lambda k: k[0]*k[1]))
L.append(r"""\begin{table}[H]
    \centering
    \small
    \begin{tabular}{|c|c|r|r|}
    \hline
    \textbf{$(p,t)$} & \textbf{Threads totales $p\,t$} & \textbf{Mediana [s]} & \textbf{Veces más lento que la mejor} \\
    \hline
""" + rows + r"""
    \hline
    \end{tabular}
    \caption{Oversubscription provocada en el computador 2 con OpenBLAS: combinaciones con $p\,t>10$ frente a la mejor combinación válida.}
    \label{tab:anexo-oversub}
\end{table}
""")
(OUT / "tablas_anexo_openblas.tex").write_text("\n".join(L), encoding="utf-8")

print("mejor OpenBLAS:", best_ob, round(I_ob[best_ob], 3), "| mejor Accelerate:", min(I_ac, key=I_ac.get), round(min(I_ac.values()), 3))
print("fila p=1 OpenBLAS:", {t: round(I_ob[(1, t)], 2) for t in ts})
print("(e) threads por worker:", e_threads)
print("oversub medianas:", {k: round(v, 1) for k, v in ov_med.items()})
print("archivos:", sorted(p.name for p in OUT.iterdir()))
