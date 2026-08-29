import numpy as np
from data_generation import generate_data
from bs_numpy import bootstrap_numpy

# 1. Generate the exact same dataset
X, y, beta_star = generate_data(seed=1111, N=10_000, k=300)

# 2. Run your fastest implementation
lb, ub, t_exec = bootstrap_numpy(X, y, B=48, p=1, base_seed=1234)

# 3. Check how many true coefficients fall within the intervals
inside_interval = (beta_star >= lb) & (beta_star <= ub)
coverage_percentage = np.mean(inside_interval) * 100

print(f"Confidence Interval Coverage: {coverage_percentage:.2f}%")