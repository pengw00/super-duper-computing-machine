import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 模型定义（必须和训练时一样）
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(10 * 12 * 12, 10)

    def forward(self, x):
        x = torch.relu(self.pool(self.conv1(x)))
        x = x.view(-1, 10 * 12 * 12)
        return self.fc(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== 第一步：训练并保存模型 ======
print("📝 训练模型中...")
from torch.optim import SGD
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = SimpleCNN().to(device)
optimizer = SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

model.train()
for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    loss = criterion(model(data), target)
    loss.backward()
    optimizer.step()

# 保存模型权重
torch.save(model.state_dict(), "mnist_cnn.pth")
print("💾 模型已保存到 mnist_cnn.pth")

# ====== 第二步：加载模型做推理 ======
print("\n🔮 加载模型进行推理...")

# 1. 创建空模型 → 加载训练好的权重
inference_model = SimpleCNN().to(device)
inference_model.load_state_dict(torch.load("mnist_cnn.pth"))
inference_model.eval()  # ⚠️ 关键：切换到推理模式（关闭 dropout/batchnorm 等）

# 2. 加载测试集
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000)

# 3. 推理：不需要计算梯度
correct = 0
total = 0
with torch.no_grad():  # ⚠️ 关键：推理时不算梯度，省显存省时间
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = inference_model(data)           # 前向传播得到预测
        pred = output.argmax(dim=1)              # 取概率最大的那个数字
        correct += (pred == target).sum().item()
        total += target.size(0)

print(f"✅ 测试集准确率: {correct}/{total} = {100.*correct/total:.1f}%")

# 4. 展示几个具体预测
data, target = next(iter(DataLoader(test_dataset, batch_size=10, shuffle=True)))
data = data.to(device)
with torch.no_grad():
    pred = inference_model(data).argmax(dim=1)

print(f"\n📊 随机10张图片的预测结果:")
print(f"   预测: {pred.tolist()}")
print(f"   真实: {target.tolist()}")
print(f"   {'✅ 全对!' if (pred.cpu() == target).all() else '❌ 有错误'}")
