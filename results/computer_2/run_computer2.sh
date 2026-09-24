#!/bin/zsh
cd /Users/joseignaciosalas/personal/Tarea-1-IIC3533
PY=/Users/joseignaciosalas/personal/Tarea-1-IIC3533/.venv/bin/python
echo "=== INICIO $(date)"
echo "=== experiment_f" && $PY -W ignore src/experiment_f.py --output-dir results/computer_2 --repetitions 5 &&
echo "=== experiment_g" && $PY -W ignore src/experiment_g.py --input results/computer_2/timings_f.csv --output-dir results/computer_2 &&
echo "=== experiment_h" && $PY -W ignore src/experiment_h.py --input results/computer_2/metrics_g.csv --output-dir results/computer_2 &&
echo "=== experiment_i" && $PY -W ignore src/experiment_i.py --output-dir results/computer_2 --repetitions 5 &&
echo "=== FIN $(date)" || echo "=== FALLO $(date)"
