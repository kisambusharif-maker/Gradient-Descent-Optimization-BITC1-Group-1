import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Simple 1-D example – pure calculus
# ============================================================
# Loss: L(w) = (w - 3)^2
# Derivative (calculus): dL/dw = 2*(w - 3)

def loss_1d(w):
    return (w - 3)**2

def grad_1d(w):          # ← this is the derivative
    return 2 * (w - 3)

w = 10.0                 # starting point
lr = 0.1
history = []

for i in range(30):
    g = grad_1d(w)       # compute gradient using calculus
    w = w - lr * g       # gradient descent update
    history.append(loss_1d(w))

print("1-D final weight ≈", w)
print("1-D final loss   ≈", loss_1d(w))


# ============================================================
# 2. Tiny neural network with MANUAL back-propagation
#    (explicit chain rule – this is the real calculus part)
# ============================================================
# Network:  input x → linear → ReLU → linear → output
# Loss:     Mean Squared Error

np.random.seed(42)

# Toy data: y = 2x + 1 + noise
X = np.linspace(-2, 2, 50).reshape(-1, 1)
y = 2 * X + 1 + 0.3 * np.random.randn(*X.shape)

# Network parameters
W1 = np.random.randn(1, 8) * 0.5   # input → hidden
b1 = np.zeros((1, 8))
W2 = np.random.randn(8, 1) * 0.5   # hidden → output
b2 = np.zeros((1, 1))

lr = 0.05
losses = []

def relu(z):
    return np.maximum(0, z)

def relu_grad(z):          # derivative of ReLU
    return (z > 0).astype(float)

for epoch in range(200):
    # ---------- Forward pass ----------
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    pred = z2

    # Loss
    loss = np.mean((pred - y)**2)
    losses.append(loss)

    # ---------- Backward pass (chain rule) ----------
    # dL/dpred
    dL_dpred = 2 * (pred - y) / len(X)

    # Layer 2 gradients
    dL_dW2 = a1.T @ dL_dpred
    dL_db2 = np.sum(dL_dpred, axis=0, keepdims=True)
    dL_da1 = dL_dpred @ W2.T

    # Layer 1 gradients (chain rule through ReLU)
    dL_dz1 = dL_da1 * relu_grad(z1)
    dL_dW1 = X.T @ dL_dz1
    dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

    # ---------- Gradient descent update ----------
    W2 -= lr * dL_dW2
    b2 -= lr * dL_db2
    W1 -= lr * dL_dW1
    b1 -= lr * dL_db1

print("\nManual backprop final loss ≈", losses[-1])


# ============================================================
# 3. Same network with PyTorch (automatic differentiation)
# ============================================================
try:
    import importlib

    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")

    # Convert data
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    model = nn.Sequential(
        nn.Linear(1, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    criterion = nn.MSELoss()

    torch_losses = []
    for epoch in range(200):
        pred = model(X_t)
        loss = criterion(pred, y_t)
        torch_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()          # ← PyTorch applies the chain rule for us
        optimizer.step()

    print("PyTorch final loss ≈", torch_losses[-1])
except ImportError:
    print("\nPyTorch not installed – skipping automatic differentiation example.")


# ============================================================
# Visualization
# ============================================================
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(history)
plt.title("1-D Gradient Descent")
plt.xlabel("Iteration")
plt.ylabel("Loss")

plt.subplot(1, 3, 2)
plt.plot(losses)
plt.title("Manual Backprop (Tiny NN)")
plt.xlabel("Epoch")
plt.ylabel("Loss")

if 'torch_losses' in locals():
    plt.subplot(1, 3, 3)
    plt.plot(torch_losses)
    plt.title("PyTorch Autograd")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/results.png", dpi=150)
plt.show()