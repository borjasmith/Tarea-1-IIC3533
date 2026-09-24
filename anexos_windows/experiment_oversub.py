"""Demostración de oversubscription: combinaciones (p,t) con p*t > p_max, versión NumPy.
Reutiliza el modo worker de experiment_i.py (un intérprete nuevo por medición)."""
import argparse, csv, json, subprocess, sys, statistics as st
from pathlib import Path

COMBOS = [(2, 10, 3), (4, 5, 3), (5, 10, 3), (10, 5, 3), (10, 10, 2)]  # (p, t, repeticiones)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output-dir", required=True); a = ap.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    worker = Path(__file__).with_name("experiment_i.py")
    rows = []
    for p, t, reps in COMBOS:
        for r in range(1, reps + 1):
            done = subprocess.run([sys.executable, str(worker), "--worker-p", str(p), "--worker-t", str(t)],
                                  check=True, capture_output=True, text=True)
            line = next(l for l in done.stdout.splitlines() if l.startswith("RESULT_JSON="))
            res = json.loads(line.removeprefix("RESULT_JSON="))
            rows.append({"p": p, "t": t, "p_times_t": p * t, "repetition": r, "time_s": res["time_s"],
                         "visible_threadpools": json.dumps(res["visible_threadpools"])})
            print(f"p={p}, t={t}, rep={r}: {res['time_s']:.3f} s", flush=True)
    with (out / "timings_oversub.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n"); w.writeheader(); w.writerows(rows)
    for p, t, _ in COMBOS:
        v = [r["time_s"] for r in rows if r["p"] == p and r["t"] == t]
        print(f"mediana p={p}, t={t} (p*t={p*t}): {st.median(v):.3f} s")

if __name__ == "__main__":
    main()
