from data_generation import generate_data
import numpy as np
import time
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression

def bootstrap_auto_v1(X, y, B, p):
    base_model = LinearRegression() 
    
    bagging_model = BaggingRegressor(
        estimator=base_model,
        n_estimators=B,
        n_jobs=p,
    )
    
    start_time = time.time()
    bagging_model.fit(X, y)
    
    coefs = []
    for est in bagging_model.estimators_:
        coefs.append(est.coef_)
        
    coefs = np.array(coefs)
    
    lower_bound = np.percentile(coefs, 2.5, axis=0)
    upper_bound = np.percentile(coefs, 97.5, axis=0)

    exec_time = time.time() - start_time
    return lower_bound, upper_bound, exec_time

def bootstrap_auto_v2(X, y, B, p):
    base_model = LinearRegression(fit_intercept=False) 
    
    bagging_model = BaggingRegressor(
        estimator=base_model,
        n_estimators=B,
        n_jobs=p,
    )
    
    start_time = time.time()
    bagging_model.fit(X, y)
    
    coefs = []
    for est in bagging_model.estimators_:
        coefs.append(est.coef_)
        
    coefs = np.array(coefs)
    
    lower_bound = np.percentile(coefs, 2.5, axis=0)
    upper_bound = np.percentile(coefs, 97.5, axis=0)

    exec_time = time.time() - start_time
    return lower_bound, upper_bound, exec_time

def bootstrap_auto_v3(X, y, B, p):
    base_model = LinearRegression(fit_intercept=False) 
    
    bagging_model = BaggingRegressor(
        estimator=base_model,
        n_estimators=B,
        n_jobs=p,
    )
    
    start_time = time.time()
    bagging_model.fit(X, y)
    
    coefs = np.array([est.coef_ for est in bagging_model.estimators_])
    
    lower_bound = np.percentile(coefs, 2.5, axis=0)
    upper_bound = np.percentile(coefs, 97.5, axis=0)

    exec_time = time.time() - start_time
    return lower_bound, upper_bound, exec_time

def bootstrap_auto_v4(X, y, B, p):
    base_model = LinearRegression(fit_intercept=False) 
    
    bagging_model = BaggingRegressor(
        estimator=base_model,
        n_estimators=B,
        n_jobs=p,
        bootstrap=True,
    )
    
    start_time = time.time()
    bagging_model.fit(X, y)
    
    coefs = np.array([est.coef_ for est in bagging_model.estimators_])
    
    lower_bound = np.percentile(coefs, 2.5, axis=0)
    upper_bound = np.percentile(coefs, 97.5, axis=0)

    exec_time = time.time() - start_time
    return lower_bound, upper_bound, exec_time

def bootstrap_auto_v5(X, y, B, p):
    base_model = LinearRegression(fit_intercept=False) 
    
    bagging_model = BaggingRegressor(
        estimator=base_model,
        n_estimators=B,
        n_jobs=p,
        bootstrap=True,
    )
    
    start_time = time.time()
    bagging_model.fit(X, y)
    
    coefs = []
    for est in bagging_model.estimators_:
        coefs.append(est.coef_)
        
    coefs = np.array(coefs)
    
    lower_bound = np.percentile(coefs, 2.5, axis=0)
    upper_bound = np.percentile(coefs, 97.5, axis=0)

    exec_time = time.time() - start_time
    return lower_bound, upper_bound, exec_time

if __name__ == "__main__":
    X, y, beta_star = generate_data(seed=1111, N=10_000, k=300)
    
    # lb, ub, t_exec = bootstrap_auto_v1(X, y, B=48, p=1)
    # lb, ub, t_exec = bootstrap_auto_v2(X, y, B=48, p=1)
    # lb, ub, t_exec = bootstrap_auto_v3(X, y, B=48, p=1)
    lb, ub, t_exec = bootstrap_auto_v4(X, y, B=48, p=1)
    # lb, ub, t_exec = bootstrap_auto_v5(X, y, B=48, p=1)
    
    print(f"Execution time (p=1): {t_exec:.4f} seconds")
    print(f"Lower bound shape: {lb.shape}")
    print(f"Upper bound shape: {ub.shape}")


# Version 1:
# Execution time (p=1): 4.7474 seconds

# Version 2:
# Execution time (p=1): 5.1836 seconds

# Version 3:
# Execution time (p=1): 5.3599 seconds

# Version 4:
# Execution time (p=1): 4.2048 seconds

# Version 5:
# Execution time (p=1): 4.1570 seconds