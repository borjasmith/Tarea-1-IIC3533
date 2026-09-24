"""Repite las mediciones de f contaminadas por el reposo del sistema y regenera f, g, h."""
import csv, json, subprocess, sys
from pathlib import Path
sys.path.insert(0, "src")
from experiment_f import save_plot  # noqa: E402

PY = sys.executable
OUT = Path("results/computer_2")
CONTAMINATED = [  # (implementación, p, repetición) — coinciden con reposos del sistema
    ("sklearn + joblib", 7, 5),
    ("sklearn + joblib", 1, 1),
    ("BaggingRegressor", 4, 5),
]

rows = list(csv.DictReader((OUT / "timings_f.csv").open()))
for row in rows:
    row["p"] = int(row["p"]); row["repetition"] = int(row["repetition"]); row["time_s"] = float(row["time_s"])

notes = []
for name, p, rep in CONTAMINATED:
    target = next(r for r in rows if r["implementation"] == name and r["p"] == p and r["repetition"] == rep)
    old = target["time_s"]
    done = subprocess.run([PY, "-W", "ignore", "src/experiment_f.py", "--worker-implementation", name,
                           "--worker-p", str(p)], check=True, capture_output=True, text=True)
    line = next(l for l in done.stdout.splitlines() if l.startswith("RESULT_JSON="))
    new = json.loads(line.removeprefix("RESULT_JSON="))["time_s"]
    target["time_s"] = new
    msg = f"{name}, p={p}, repetición={rep}: {old:.3f} s -> {new:.3f} s"
    print(msg, flush=True); notes.append(msg)

with (OUT / "timings_f.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["implementation", "p", "repetition", "time_s"], lineterminator="\n")
    w.writeheader(); w.writerows(rows)
save_plot(rows, OUT / "tiempos_f.png", max(r["p"] for r in rows))

(OUT / "NOTAS.md").write_text(
    "# Notas de la corrida del computador 2\n\n"
    "- Corrida original de f: 2026-09-22 12:43 a 13:47, con el equipo a batería. El sistema entró en reposo "
    "tres veces (13:06-13:08, 13:12-13:20, 13:26-13:43) y el cronómetro siguió corriendo, contaminando "
    "exactamente tres mediciones. Se repitieron con el equipo enchufado y `caffeinate` activo:\n"
    + "".join(f"  - {m}\n" for m in notes)
    + "- Los experimentos i, g y h se ejecutaron o recalcularon con reposo bloqueado.\n"
    "- Comandos: ver `run_computer2.sh` (f, g, h, i en cadena) y `patch_f.py`.\n",
    encoding="utf-8",
)
print("timings_f.csv y tiempos_f.png actualizados", flush=True)
