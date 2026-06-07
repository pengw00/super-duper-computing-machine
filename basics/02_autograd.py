"""
02 - Autograd
=============
PyTorch tracks every operation on tensors that have `requires_grad=True`
and builds a computation graph.  Calling `.backward()` walks the graph in
reverse to compute gradients automatically (reverse-mode autodiff).

Run:
    python basics/02_autograd.py
"""

import torch

# ------------------------------------------------------------------
# 1. Basic gradient computation
# ------------------------------------------------------------------
# y = x^2  ->  dy/dx = 2x
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
y.backward()
print(f"x = {x.item()},  y = x² = {y.item()},  dy/dx = {x.grad.item()}")  # 6.0

# ------------------------------------------------------------------
# 2. Computational graph with multiple ops
# ------------------------------------------------------------------
# z = (a + b) * c
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
c = torch.tensor(4.0, requires_grad=True)

z = (a + b) * c
z.backward()
print(f"\ndz/da = {a.grad.item()}")  # c = 4
print(f"dz/db = {b.grad.item()}")  # c = 4
print(f"dz/dc = {c.grad.item()}")  # a + b = 5

# ------------------------------------------------------------------
# 3. Disabling gradient tracking
# ------------------------------------------------------------------
x = torch.randn(3, requires_grad=True)

# torch.no_grad() — cheapest way to disable tracking temporarily
with torch.no_grad():
    y = x * 2
print(f"\ny.requires_grad inside no_grad: {y.requires_grad}")  # False

# .detach() — creates a view that shares storage but has no grad
z = x.detach()
print(f"z.requires_grad after detach: {z.requires_grad}")       # False

# ------------------------------------------------------------------
# 4. Gradient accumulation and zeroing
# ------------------------------------------------------------------
w = torch.tensor(1.0, requires_grad=True)
for step in range(3):
    loss = (w * 2) ** 2
    loss.backward()
    print(f"step {step}: grad = {w.grad.item()}")  # accumulates!

w.grad.zero_()  # reset before next backward pass (optimizers do this)
print(f"after zero_(): grad = {w.grad.item()}")

# ------------------------------------------------------------------
# 5. A tiny gradient-descent step by hand
# ------------------------------------------------------------------
# Minimise f(x) = (x - 5)^2
x = torch.tensor(0.0, requires_grad=True)
lr = 0.1

print("\nGradient descent on f(x) = (x-5)^2:")
for i in range(20):
    f = (x - 5) ** 2
    f.backward()
    with torch.no_grad():
        x -= lr * x.grad
    x.grad.zero_()
    if i % 5 == 4:
        print(f"  step {i+1:2d}: x = {x.item():.4f}, f(x) = {f.item():.4f}")

print(f"Converged to x ≈ {x.item():.4f}  (true minimum: 5.0)")
