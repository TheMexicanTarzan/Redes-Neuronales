import numpy as np
from sklearn.svm import SVC

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

# ---------- Datos ----------
X_tr, y_tr = two_moons(300,  d=-6.5, seed=1)
X_te, y_te = two_moons(2000, d=-6.5, seed=2)

# ---------- SVM con kernel gaussiano ----------
# k(x, x') = exp(-gamma * ||x - x'||^2);  gamma = 1 / (2 * sigma^2).
# Tomamos sigma = 1 (heuristica de Haykin) -> gamma = 0.5.
# C grande (1e3) ~ "C = infinito" del experimento del libro.
sigma  = 1.0
gamma  = 1.0 / (2.0 * sigma**2)
clf    = SVC(kernel="rbf", gamma=gamma, C=1e3).fit(X_tr, y_tr)

# ---------- Evaluacion ----------
err_tr = np.mean(clf.predict(X_tr) != y_tr) * 100
err_te = np.mean(clf.predict(X_te) != y_te) * 100
nsv    = clf.support_.size

print(f"Numero de vectores de soporte : {nsv} / {len(X_tr)}")
print(f"Error de entrenamiento        : {err_tr:.2f}%")
print(f"Error de prueba               : {err_te:.2f}%  ({int(err_te*20)} errores de 2000)")

# ---------- Frontera de decision (opcional, para graficar) ----------
def decision_surface(clf, x_range=(-15, 25), y_range=(-10, 15), n=300):
    xx, yy = np.meshgrid(np.linspace(*x_range, n), np.linspace(*y_range, n))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    return xx, yy, Z

# ---------- Comparacion con C pequeno ----------
for C in [1e-1, 1.0, 10.0, 1e3]:
    m = SVC(kernel="rbf", gamma=gamma, C=C).fit(X_tr, y_tr)
    e = np.mean(m.predict(X_te) != y_te) * 100
    print(f"C = {C:7.1f}  ->  vectores de soporte = {m.support_.size:3d}, "
          f"error de prueba = {e:5.2f}%")