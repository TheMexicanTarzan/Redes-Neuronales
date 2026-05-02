import numpy as np
import matplotlib.pyplot as plt

# ---------- Generador de las dos lunas ----------
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

# ---------- Perceptrón de Rosenblatt ----------
class Perceptron:
    """Perceptrón con tasa de aprendizaje decreciente linealmente."""
    def __init__(self, n_features, eta_max=0.1, eta_min=1e-5, epochs=50):
        self.w  = np.zeros(n_features + 1)   # incluye sesgo
        self.eta_max, self.eta_min = eta_max, eta_min
        self.epochs = epochs
        self.mse_history = []

    def _augment(self, X):
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def fit(self, X, y):
        Xa = self._augment(X)
        for ep in range(self.epochs):
            eta = self.eta_max + (self.eta_min - self.eta_max) * ep / self.epochs
            errs = []
            for xi, yi in zip(Xa, y):
                y_hat = np.sign(self.w @ xi) or 1.0     # signo (no cero)
                err   = yi - y_hat
                self.w += eta * err * xi
                errs.append(0.5 * err**2)
            self.mse_history.append(np.mean(errs))
        return self

    def predict(self, X):
        return np.sign(self._augment(X) @ self.w)

# ---------- Ejecución del experimento ----------
if __name__ == "__main__":
    for d in [1.0, -4.0]:                   # caso separable y no separable
        X_tr, y_tr = two_moons(1000, d=d, seed=1)
        X_te, y_te = two_moons(2000, d=d, seed=2)
        clf = Perceptron(n_features=2, epochs=50).fit(X_tr, y_tr)
        err = np.mean(clf.predict(X_te) != y_te)
        print(f"d = {d:+.1f}  ->  error de clasificacion = {err*100:.2f}%")
