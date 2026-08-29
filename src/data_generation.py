import numpy as np

def generate_data(seed=1111, N=10_000, k=300):
    rng = np.random.default_rng(seed)

    # I: coeficientes 
    beta_star = rng.standard_normal(size=(k + 1))

    # II: matriz de datos
    base_matrix = rng.standard_normal(size=(N, k))
    ones_column = np.ones((N, 1))
    X = np.hstack((ones_column, base_matrix))

    # III: calculo
    noise = rng.standard_normal(size=N)
    y = X @ beta_star + noise

    return X, y, beta_star

if __name__ == "__main__":
    N = 10_000
    k = 300
    B = 48
    seed = 1111

    X, y, beta_star = generate_data(seed, N, k)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"beta_star shape: {beta_star.shape}")