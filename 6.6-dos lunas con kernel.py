import numpy as np

def two_moons(n_samples=1000, r=10.0, w=6.0, d=1.0, seed=0):
    """Genera n_samples por clase del problema de las dos lunas."""
    rng = np.random.default_rng(seed)
    # Luna A: semicírculo superior centrado en el origen
    theta_A = rng.uniform(0, np.pi, n_samples)
    rad_A   = rng.uniform(r - w/2, r + w/2, n_samples)
    XA = np.column_stack([rad_A * np.cos(theta_A),
                          rad_A * np.sin(theta_A)])
    # Luna B: semicírculo inferior, desplazado (r, -d)
    theta_B = rng.uniform(np.pi, 2*np.pi, n_samples)
    rad_B   = rng.uniform(r - w/2, r + w/2, n_samples)
    XB = np.column_stack([rad_B * np.cos(theta_B) + r,
                          rad_B * np.sin(theta_B) - d])
    X = np.vstack([XA, XB])
    y = np.concatenate([ np.ones(n_samples), -np.ones(n_samples)])
    idx = rng.permutation(len(X))
    return X[idx], y[idx]

# ---------- Etapa 1a: K-means (algoritmo de Lloyd) ----------
def kmeans(X, K, n_iter=100, seed=0):
    rng = np.random.default_rng(seed)
    mu = X[rng.choice(len(X), K, replace=False)].copy()    # init aleatorio
    for _ in range(n_iter):
        d2 = np.sum((X[:, None, :] - mu[None, :, :])**2, axis=2)
        labels = np.argmin(d2, axis=1)
        new_mu = np.array([X[labels == j].mean(axis=0)
                           if np.any(labels == j) else mu[j]
                           for j in range(K)])
        if np.allclose(new_mu, mu):
            break
        mu = new_mu
    return mu

# ---------- Etapa 1b: ancho comun (Eq. 5.49) ----------
def gaussian_features(X, mu, sigma):
    """phi_j(x) = exp(-||x - mu_j||^2 / (2*sigma^2))."""
    d2 = np.sum((X[:, None, :] - mu[None, :, :])**2, axis=2)
    return np.exp(-d2 / (2.0 * sigma**2))

# ---------- Etapa 2: RLS (Eqs. 5.45, 5.47, 5.38, 5.48) ----------
class RLS:
    def __init__(self, K, delta=1e-2):
        self.w = np.zeros(K)
        self.P = np.eye(K) / delta            # P(0) = delta^-1 * I
        self.mse_history = []

    def fit(self, Phi, y, n_epochs=10):
        for ep in range(n_epochs):
            sq_err = 0.0
            for phi_n, d_n in zip(Phi, y):
                Pphi = self.P @ phi_n
                xi   = d_n - self.w @ phi_n            # error a priori
                g    = Pphi / (1.0 + phi_n @ Pphi)     # ganancia
                self.w += g * xi
                self.P -= np.outer(g, Pphi)
                sq_err += xi * xi
            self.mse_history.append(sq_err / len(y))
        return self

# ---------- Ejecucion completa ----------
if __name__ == "__main__":
    K = 20
    X_tr, y_tr = two_moons(1000, d=-6.0, seed=1)
    X_te, y_te = two_moons(2000, d=-6.0, seed=2)

    # Etapa 1: K-means y ancho comun
    mu = kmeans(X_tr, K, seed=0)
    d_max = np.max(np.linalg.norm(mu[:, None] - mu[None, :], axis=2))
    sigma = d_max / np.sqrt(2.0 * K)
    print(f"d_max = {d_max:.3f}, sigma = {sigma:.3f}")

    # Mapeo a espacio de caracteristicas
    Phi_tr = gaussian_features(X_tr, mu, sigma)
    Phi_te = gaussian_features(X_te, mu, sigma)

    # Etapa 2: RLS
    rls = RLS(K=K).fit(Phi_tr, y_tr, n_epochs=10)
    pred = np.sign(Phi_te @ rls.w)
    err  = np.mean(pred != y_te)
    print(f"Error de clasificacion en test: {err*100:.2f}%")
    print(f"MSE final de entrenamiento    : {rls.mse_history[-1]:.4f}")
