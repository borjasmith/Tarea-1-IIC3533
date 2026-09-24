# Notas de la corrida del computador 2

- Corrida original de f: 2026-09-22 12:43 a 13:47, con el equipo a batería. El sistema entró en reposo tres veces (13:06-13:08, 13:12-13:20, 13:26-13:43) y el cronómetro siguió corriendo, contaminando exactamente tres mediciones. Se repitieron con el equipo enchufado y `caffeinate` activo:
  - sklearn + joblib, p=7, repetición=5: 123.935 s -> 15.945 s
  - sklearn + joblib, p=1, repetición=1: 498.480 s -> 29.042 s
  - BaggingRegressor, p=4, repetición=5: 1065.475 s -> 16.945 s
- Los experimentos i, g y h se ejecutaron o recalcularon con reposo bloqueado.
- Comandos: ver `run_computer2.sh` (f, g, h, i en cadena) y `patch_f.py`.
