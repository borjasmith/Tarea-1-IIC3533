import numpy as np
from data_generation import generate_data
from bs_auto import bootstrap_auto_v4
from bs_sklearn import bootstrap_sklearn_v3
from bs_numpy import bootstrap_numpy_v2

X, y, beta_star = generate_data(seed=1111, N=10_000, k=300)

# Ejecutar las tres versiones (usando p=1 para la prueba de correctitud)
lb_auto, ub_auto, _ = bootstrap_auto_v4(X, y, B=48, p=1)
lb_skl, ub_skl, _ = bootstrap_sklearn_v3(X, y, B=48, p=1, base_seed=1234)
lb_np, ub_np, _ = bootstrap_numpy_v2(X, y, B=48, p=1, base_seed=1234)

# 1. Consistencia con beta_star (Cobertura)
cov_auto = np.mean((beta_star >= lb_auto) & (beta_star <= ub_auto)) * 100
cov_skl = np.mean((beta_star >= lb_skl) & (beta_star <= ub_skl)) * 100
cov_np = np.mean((beta_star >= lb_np) & (beta_star <= ub_np)) * 100

print(f"Cobertura Auto: {cov_auto:.2f}%")
print(f"Cobertura Sklearn: {cov_skl:.2f}%")
print(f"Cobertura Numpy: {cov_np:.2f}%")

# 2. Equivalencia entre versiones
# Tolerancia de punto flotante para comparar arreglos
is_identical = np.allclose(lb_skl, lb_np) and np.allclose(ub_skl, ub_np)
print(f"Sklearn y Numpy son matemáticamente idénticos: {is_identical}")


# Cobertura Auto: 93.69%
# Cobertura Sklearn: 91.69%
# Cobertura Numpy: 91.69%
# Sklearn y Numpy son matemáticamente idénticos: True