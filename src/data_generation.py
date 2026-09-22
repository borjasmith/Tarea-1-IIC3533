import numpy as np

# Parámetros del problema (enunciado actualizado: N pasó de 10 000 a 100 000).
SEED = 1111
N_OBS = 100_000
K_VARS = 300
B_RESAMPLES = 48


def generate_data(seed=SEED, N=N_OBS, k=K_VARS):
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-obs", type=int, default=N_OBS)
    args = parser.parse_args()

    X, y, beta_star = generate_data(SEED, args.n_obs, K_VARS)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"beta_star shape: {beta_star.shape}")