import numpy as np
import time
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from data_generation import generate_data

def fit_resample_v1(X, y):
    N = X.shape[0]
    indices = np.random.choice(N, size=N, replace=True)
    
    model = LinearRegression()
    model.fit(X[indices], y[indices])
    
    return model.coef_

def bootstrap_sklearn_v1(X, y, B, p):
    start_time = time.time()
    
    coefs = Parallel(n_jobs=p)(
        delayed(fit_resample_v1)(X, y) for _ in range(B)
    )
    
    coefs = np.array(coefs)
    lower_bound = np.percentile(coefs, 2.5, axis=0)
    upper_bound = np.percentile(coefs, 97.5, axis=0)
    
    exec_time = time.time() - start_time
    
    return lower_bound, upper_bound, exec_time

if __name__ == "__main__":
    X, y, beta_star = generate_data(seed=1111, N=10_000, k=300)
    
    lb, ub, t_exec = bootstrap_sklearn_v1(X, y, B=48, p=1)
    
    print(f"Execution time (p=1): {t_exec:.4f} seconds")
    print(f"Lower bound shape: {lb.shape}")
    print(f"Upper bound shape: {ub.shape}")


# Version 1:
# Execution time (p=1): 5.9958 seconds