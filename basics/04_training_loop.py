"""
04 - Full Training Loop
========================
Putting it all together: dataset, dataloader, model, optimizer, and the
classic train / evaluate cycle.  We train a small MLP on the MNIST
handwritten-digit dataset.

Run:
    python basics/04_training_loop.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ------------------------------------------------------------------
# 0. Reproducibility
# ------------------------------------------------------------------
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ------------------------------------------------------------------
# 1. Data
# ------------------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),   # MNIST mean & std
])

train_dataset = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=256, shuffle=False, num_workers=0)

print(f"Train samples: {len(train_dataset)},  Test samples: {len(test_dataset)}")

# ------------------------------------------------------------------
# 2. Model
# ------------------------------------------------------------------
class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


model = DigitClassifier().to(device)
print(model)

# ------------------------------------------------------------------
# 3. Optimiser and loss
# ------------------------------------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ------------------------------------------------------------------
# 4. Training and evaluation helpers
# ------------------------------------------------------------------

def train_one_epoch(epoch: int) -> float:
    model.train()
    total_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()           # 1. reset gradients
        logits = model(images)          # 2. forward pass
        loss = criterion(logits, labels)# 3. compute loss
        loss.backward()                 # 4. backward pass
        optimizer.step()                # 5. update weights

        total_loss += loss.item()
    return total_loss / len(train_loader)


@torch.no_grad()
def evaluate() -> tuple[float, float]:
    model.eval()
    total_loss, correct = 0.0, 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / len(test_dataset)
    return avg_loss, accuracy


# ------------------------------------------------------------------
# 5. Training loop
# ------------------------------------------------------------------
NUM_EPOCHS = 3

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss = train_one_epoch(epoch)
    test_loss, test_acc = evaluate()
    print(
        f"Epoch {epoch}/{NUM_EPOCHS}  "
        f"train_loss={train_loss:.4f}  "
        f"test_loss={test_loss:.4f}  "
        f"test_acc={test_acc*100:.2f}%"
    )

print("\nDone!")
