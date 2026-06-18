"""
03 - Neural Networks with nn.Module
====================================
`torch.nn` provides composable building blocks for neural networks.
Every layer, loss function, and the network itself is a subclass of
`nn.Module`.

Run:
    python basics/03_neural_network.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------
# 1. Building a simple Multi-Layer Perceptron (MLP)
# ------------------------------------------------------------------
class MLP(nn.Module):
    """A fully-connected network: input -> hidden -> output."""

    def __init__(self, in_features: int, hidden: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))   # hidden layer with ReLU activation
        x = self.fc2(x)            # output layer (no activation — raw logits)
        return x


model = MLP(in_features=4, hidden=8, out_features=3)
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal params: {total_params}, Trainable: {trainable_params}")

# ------------------------------------------------------------------
# 2. Forward pass
# ------------------------------------------------------------------
x = torch.randn(5, 4)   # batch of 5 samples, each with 4 features
logits = model(x)
print(f"\nInput shape:  {x.shape}")
print(f"Output shape: {logits.shape}")   # (5, 3)

# ------------------------------------------------------------------
# 3. Loss functions
# ------------------------------------------------------------------
targets = torch.tensor([0, 1, 2, 1, 0])   # class indices

# Cross-entropy loss (combines LogSoftmax + NLLLoss)
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, targets)
print(f"\nCross-entropy loss: {loss.item():.4f}")

# ------------------------------------------------------------------
# 4. Using nn.Sequential for simple architectures
# ------------------------------------------------------------------
simple = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 3),
)
print("\nSequential model:")
print(simple)

# ------------------------------------------------------------------
# 5. Saving and loading model weights
# ------------------------------------------------------------------
import tempfile, os

with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
    path = f.name

torch.save(model.state_dict(), path)
print(f"\nSaved weights to {path}")

new_model = MLP(in_features=4, hidden=8, out_features=3)
new_model.load_state_dict(torch.load(path, weights_only=True))
new_model.eval()
print("Loaded weights into new_model.")

os.unlink(path)

# ------------------------------------------------------------------
# 6. Common activation functions
# ------------------------------------------------------------------
x = torch.linspace(-3, 3, 7)
print("\nx:        ", x.numpy().round(2))
print("ReLU(x):  ", F.relu(x).numpy().round(2))
print("Sigmoid:  ", torch.sigmoid(x).numpy().round(2))
print("Tanh:     ", torch.tanh(x).numpy().round(2))
print("GELU:     ", F.gelu(x).numpy().round(2))
