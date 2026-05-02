import numpy as np
import torch
import torch.nn as nn

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
torch.manual_seed(0)
X_tr_np, y_tr_np = two_moons(1000, d=-4.0, seed=1)
X_te_np, y_te_np = two_moons(2000, d=-4.0, seed=2)

X_tr = torch.tensor(X_tr_np, dtype=torch.float32)
y_tr = torch.tensor(y_tr_np, dtype=torch.float32).unsqueeze(1)
X_te = torch.tensor(X_te_np, dtype=torch.float32)
y_te = torch.tensor(y_te_np, dtype=torch.float32).unsqueeze(1)

# ---------- Arquitectura: 2 - 20 - 1 con tanh ----------
class MLP(nn.Module):
    def __init__(self, n_in=2, n_hidden=20, n_out=1):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)
        # Inicializacion LeCun: var(W) = 1 / fan_in
        nn.init.normal_(self.fc1.weight, std=(1.0 / n_in) ** 0.5)
        nn.init.normal_(self.fc2.weight, std=(1.0 / n_hidden) ** 0.5)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        h = torch.tanh(self.fc1(x))
        return self.fc2(h)

# ---------- Entrenamiento on-line con eta atenuado linealmente ----------
def train_mlp(model, X, y, n_epochs=50, eta_max=1e-1, eta_min=1e-5):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=eta_max, momentum=0.9)
    mse_history = []
    n = X.shape[0]
    for epoch in range(n_epochs):
        # Annealing lineal
        eta = eta_max + (eta_min - eta_max) * epoch / n_epochs
        for g in optimizer.param_groups:
            g["lr"] = eta
        # Aleatorizacion del orden de presentacion
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in perm:
            xi, yi = X[i:i+1], y[i:i+1]
            optimizer.zero_grad()
            loss = loss_fn(model(xi), yi)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        mse_history.append(epoch_loss / n)
    return mse_history

# ---------- Ejecucion ----------
model = MLP()
mse = train_mlp(model, X_tr, y_tr)

with torch.no_grad():
    pred = torch.sign(model(X_te))
    err  = (pred != y_te).float().mean().item()
print(f"Error de clasificacion en test: {err*100:.2f}%")
print(f"MSE final de entrenamiento    : {mse[-1]:.4f}")
