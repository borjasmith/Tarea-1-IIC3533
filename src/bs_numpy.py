import numpy as np
import time
from joblib import Parallel, delayed
from data_generation import generate_data

def fit_numpy_v1(X, y, seed_b):
    rng = np.random.default_rng(seed_b)
    N = X.shape[0]
    indices = rng.choice(N, size=N, replace=True)
    X_b = X[indices]
    y_b = y[indices]
    
    XT_X_inv = np.linalg.inv(X_b.T @ X_b)
    coef = XT_X_inv @ (X_b.T @ y_b)
    
    return coef

def fit_numpy_v2(X, y, seed_b):
    rng = np.random.default_rng(seed_b)
    N = X.shape[0]
    indices = rng.choice(N, size=N, replace=True)
    X_b = X[indices]
    y_b = y[indices]
    
    coef = np.linalg.solve(X_b.T @ X_b, X_b.T @ y_b)
    
    return coef

def bootstrap_numpy_v1(X, y, B, p, base_seed=1234):
    start_time = time.time()
    seeds = [base_seed + i for i in range(B)]
    
    coefs = Parallel(n_jobs=p)(
        delayed(fit_numpy_v1)(X, y, seeds[i]) for i in range(B)
    )
    
    coefs = np.array(coefs)
    lower_bound = np.percentile(coefs, 2.5, axis=0)
    upper_bound = np.percentile(coefs, 97.5, axis=0)
    
    exec_time = time.time() - start_time
    
    return lower_bound, upper_bound, exec_time

def bootstrap_numpy_v2(X, y, B, p, base_seed=1234):
    start_time = time.time()
    seeds = [base_seed + i for i in range(B)]
    
    coefs = Parallel(n_jobs=p)(
        delayed(fit_numpy_v2)(X, y, seeds[i]) for i in range(B)
    )
    
    coefs = np.array(coefs)
    lower_bound = np.percentile(coefs, 2.5, axis=0)
    upper_bound = np.percentile(coefs, 97.5, axis=0)
    
    exec_time = time.time() - start_time
    
    return lower_bound, upper_bound, exec_time

if __name__ == "__main__":
    X, y, beta_star = generate_data(seed=1111, N=10_000, k=300)
    
    # lb, ub, t_exec = bootstrap_numpy_v1(X, y, B=48, p=1)
    lb, ub, t_exec = bootstrap_numpy_v2(X, y, B=48, p=1)
    
    print(f"Execution time (p=1): {t_exec:.4f} seconds")
    print(f"Lower bound shape: {lb.shape}")
    print(f"Upper bound shape: {ub.shape}")


# Version 1:
# Execution time (p=1): 1.0937 seconds

# Version 2:
# Execution time (p=1): 1.0204 seconds