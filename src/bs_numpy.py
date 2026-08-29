import numpy as np
import time
from joblib import Parallel, delayed
from data_generation import generate_data

def fit_single_bootstrap_numpy(X, y, seed_b):
    # Initialize a unique random generator for this specific iteration
    rng = np.random.default_rng(seed_b)
    N = X.shape[0]
    
    # Sample N indices with replacement
    indices = rng.choice(N, size=N, replace=True)
    X_b = X[indices]
    y_b = y[indices]
    
    # Solve OLS using NumPy linear algebra
    # Equation: (X^T @ X) * beta = (X^T @ y)
    XT_X = X_b.T @ X_b
    XT_y = X_b.T @ y_b
    coef = np.linalg.solve(XT_X, XT_y)
    
    return coef

def bootstrap_numpy(X, y, B, p, base_seed=1234):
    # Generate B unique seeds
    seeds = [base_seed + i for i in range(B)]
    
    start_time = time.time()
    
    # Distribute the iterations across p processes
    coefs = Parallel(n_jobs=p)(
        delayed(fit_single_bootstrap_numpy)(X, y, seeds[i]) 
        for i in range(B)
    )
    
    exec_time = time.time() - start_time
    
    # Calculate 95% confidence intervals
    coefs = np.array(coefs)
    lower_bound = np.percentile(coefs, 2.5, axis=0)
    upper_bound = np.percentile(coefs, 97.5, axis=0)
    
    return lower_bound, upper_bound, exec_time

if __name__ == "__main__":
    X, y, beta_star = generate_data(seed=1111, N=10_000, k=300)
    
    lb, ub, t_exec = bootstrap_numpy(X, y, B=48, p=1)
    
    print(f"Execution time (p=1): {t_exec:.4f} seconds")
    print(f"Lower bound shape: {lb.shape}")
    print(f"Upper bound shape: {ub.shape}")