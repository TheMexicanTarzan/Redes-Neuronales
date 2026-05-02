import numpy as np

class RegularizedLeastSquares:
    """Clasificador binario por minimos cuadrados regularizados.

    Resuelve  w = (X^T X + lambda*I)^(-1) X^T d
    con d en {-1,+1}.  Para lambda=0 se obtiene la solucion ML/OLS.
    """
    def __init__(self, lam=0.0):
        self.lam = lam
        self.w   = None

    def _augment(self, X):
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def fit(self, X, y):
        Xa  = self._augment(X)
        M   = Xa.shape[1]
        A   = Xa.T @ Xa + self.lam * np.eye(M)
        b   = Xa.T @ y
        self.w = np.linalg.solve(A, b)
        return self

    def predict(self, X):
        return np.sign(self._augment(X) @ self.w)

# ---------- Experimento ----------
if __name__ == "__main__":
    from perceptron_demo import two_moons          # del listado anterior
    for d in [1.0, 0.0, -4.0]:
        Xtr, ytr = two_moons(1000, d=d, seed=1)
        Xte, yte = two_moons(2000, d=d, seed=2)
        for lam in [0.0, 1e-2, 1.0]:
            clf = RegularizedLeastSquares(lam=lam).fit(Xtr, ytr)
            err = np.mean(clf.predict(Xte) != yte) * 100
            print(f"d={d:+.1f}  lambda={lam:.2f}  ->  error = {err:5.2f}%")
