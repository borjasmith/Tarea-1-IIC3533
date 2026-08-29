import numpy as np
import time
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from data_generation import generate_data

def bootstrap_auto(X, y, B, p):
    # Initialize the base estimator without an intercept
    base_model = LinearRegression(fit_intercept=False)
    
    # Configure the bagging regressor for internal parallelism
    bagging_model = BaggingRegressor(
        estimator=base_model,
        n_estimators=B,
        n_jobs=p,
        bootstrap=True
    )
    
    # Track only the fitting time
    start_time = time.time()
    bagging_model.fit(X, y)
    exec_time = time.time() - start_time
    
    # Extract coefficients from all B estimators
    coefs = np.array([est.coef_ for est in bagging_model.estimators_])
    
    # Calculate 95% confidence intervals
    lower_bound = np.percentile(coefs, 2.5, axis=0)
    upper_bound = np.percentile(coefs, 97.5, axis=0)
    
    return lower_bound, upper_bound, exec_time

if __name__ == "__main__":
    X, y, beta_star = generate_data(seed=1111, N=10_000, k=300)
    
    # Test with 1 process
    lb, ub, t_exec = bootstrap_auto(X, y, B=48, p=1)
    
    print(f"Execution time (p=1): {t_exec:.4f} seconds")
    print(f"Lower bound shape: {lb.shape}")
    print(f"Upper bound shape: {ub.shape}")