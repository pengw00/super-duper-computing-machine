"""
01 - Tensors
============
The fundamental data structure in PyTorch.  Everything — inputs, weights,
gradients — lives in a tensor.

Run:
    python basics/01_tensors.py
"""

import torch

# ------------------------------------------------------------------
# 1. Creating tensors
# ------------------------------------------------------------------
scalar = torch.tensor(3.14)          # 0-D tensor (scalar)
vector = torch.tensor([1.0, 2.0, 3.0])  # 1-D tensor
matrix = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)  # 2-D

print("scalar:", scalar, "  shape:", scalar.shape)
print("vector:", vector, "  shape:", vector.shape)
print("matrix:\n", matrix, "\n  shape:", matrix.shape)

# Convenience constructors
zeros = torch.zeros(2, 3)
ones  = torch.ones(2, 3)
rand  = torch.rand(2, 3)        # uniform [0, 1)
randn = torch.randn(2, 3)       # standard normal
arange = torch.arange(0, 10, 2) # like range()

print("\nzeros:\n", zeros)
print("ones:\n",  ones)
print("rand:\n",  rand)
print("arange:",  arange)

# ------------------------------------------------------------------
# 2. Tensor attributes
# ------------------------------------------------------------------
t = torch.randn(3, 4)
print("\nshape:", t.shape)
print("dtype:", t.dtype)
print("device:", t.device)

# ------------------------------------------------------------------
# 3. Indexing and slicing
# ------------------------------------------------------------------
x = torch.arange(12).reshape(3, 4).float()
print("\nx:\n", x)
print("x[0]:", x[0])          # first row
print("x[:, 1]:", x[:, 1])    # second column
print("x[1:, 2:]:\n", x[1:, 2:])  # slicing

# ------------------------------------------------------------------
# 4. Common operations
# ------------------------------------------------------------------
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print("\na + b:", a + b)
print("a * b:", a * b)          # element-wise
print("a @ b:", a @ b)          # dot product
print("a.sum():", a.sum())
print("a.mean():", a.mean())
print("a.max():", a.max())

# Matrix multiplication
A = torch.randn(2, 3)
B = torch.randn(3, 4)
C = A @ B          # or torch.matmul(A, B)
print("\nA @ B shape:", C.shape)

# ------------------------------------------------------------------
# 5. Reshaping
# ------------------------------------------------------------------
t = torch.arange(24)
print("\nt shape:", t.shape)
t2 = t.reshape(2, 3, 4)
print("reshaped to (2,3,4):", t2.shape)
t3 = t2.view(-1)          # flatten
print("flattened:", t3.shape)

# ------------------------------------------------------------------
# 6. Moving to GPU (if available)
# ------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}")
t_gpu = torch.randn(3, 3).to(device)
print("tensor on device:", t_gpu.device)
