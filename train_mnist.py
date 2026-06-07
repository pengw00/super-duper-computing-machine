import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. 验证你的 A10 显卡是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 当前使用的设备: {device} (如果是 cuda，说明 A10 正常工作！)")

# 2. 准备 MNIST 手写数字数据集（自动下载）
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 3. 手写一个简单的卷积神经网络 (CNN)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # 卷积层：提取图像特征
        self.pool = nn.MaxPool2d(2)                    # 池化层：压缩图像
        self.fc = nn.Linear(10 * 12 * 12, 10)         # 全连接层：输出10个数字的概率

    def forward(self, x):
        x = torch.relu(self.pool(self.conv1(x)))      # 前向传播：激活函数 + 池化
        x = x.view(-1, 10 * 12 * 12)                  # 展平图像
        return self.fc(x)

model = SimpleCNN().to(device)

# 4. 定义损失函数（怎么算错得多离谱）和优化器（怎么修正错误）
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 5. 开始训练 (只跑 1 个轮次 Epoch 看看效果)
model.train()
for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)  # 把数据送进 A10 显卡

    optimizer.zero_grad()                # 1. 清空上一步的梯度
    output = model(data)                 # 2. 前向传播（模型做出预测）
    loss = criterion(output, target)     # 3. 计算损失（误差有多大）
    loss.backward()                      # 4. 反向传播（把误差传回去，计算每个参数该怎么改）
    optimizer.step()                     # 5. 更新参数（真正迈出修正的一步）

    if batch_idx % 200 == 0:
        print(f"训练进度: [{batch_idx * len(data)}/{len(train_loader.dataset)}] | 误差(Loss): {loss.item():.4f}")

print("🎉 训练完成！你的 A10 显卡已经成功跑通了第一个深度学习模型！")
