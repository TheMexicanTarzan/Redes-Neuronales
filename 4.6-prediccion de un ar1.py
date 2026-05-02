import numpy as np
import matplotlib.pyplot as plt

# ---------- Parametros del modelo AR(1) ----------
a_true     = 0.99
sigma_eps2 = 0.02
sigma_x2   = sigma_eps2 / (1.0 - a_true**2)   # varianza estacionaria
eta        = 1e-3
N_iter, N_runs = 5000, 100

def ar1_realization(n_iter, seed):
    """Realizacion de longitud n_iter+1 del proceso x(n) = a*x(n-1) + eps(n)."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, np.sqrt(sigma_eps2), size=n_iter + 1)
    x   = np.empty(n_iter + 1)
    x[0] = rng.normal(0.0, np.sqrt(sigma_x2))   # arranque estacionario
    for k in range(1, n_iter + 1):
        x[k] = a_true * x[k - 1] + eps[k]
    return x

def lms_scalar_predictor(x, eta=eta):
    """Predictor LMS de un parametro: w(n+1) = w(n) + eta * x(n-1) * e(n)."""
    n = len(x) - 1
    w = 0.0
    sq_err = np.empty(n)
    for k in range(1, n + 1):
        e = x[k] - w * x[k - 1]
        w += eta * x[k - 1] * e
        sq_err[k - 1] = e * e
    return sq_err

# ---------- Ensemble averaging sobre N_runs ----------
mse_runs = np.zeros((N_runs, N_iter))
for r in range(N_runs):
    x = ar1_realization(N_iter, seed=r)
    mse_runs[r] = lms_scalar_predictor(x)
mse_emp = mse_runs.mean(axis=0)

# ---------- Curva teorica: Eq. (3.63) con M = 1 ----------
n_axis = np.arange(1, N_iter + 1)
lam, v0 = sigma_x2, a_true
J_min   = sigma_eps2
J_theo  = (J_min
           + 0.5 * eta * J_min * lam
           + lam * (v0**2 - 0.5 * eta * J_min) * (1.0 - eta * lam) ** (2 * n_axis))

# ---------- Visualizacion ----------
fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(n_axis, mse_emp, lw=0.7, label="Experimento (100 corridas)")
ax.semilogy(n_axis, J_theo, lw=2.0, label="Teoria (Eq. 3.63)")
ax.set(xlabel="Iteracion n", ylabel="MSE",
       title=f"LMS sobre AR(1), eta = {eta}")
ax.legend(); ax.grid(True, which="both", alpha=0.3)
plt.tight_layout(); plt.show()

print(f"J_min teorico            = {J_min:.4f}")
print(f"MSE empirico al final    = {mse_emp[-100:].mean():.4f}")
print(f"MSE teorico al final     = {J_theo[-1]:.4f}")
print(f"Misadjustment (eta*lam/2)= {0.5*eta*lam:.5f}")
