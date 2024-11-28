import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 标准化
])
# 像素值的归一化：在使用 ToTensor() 时，像素值已经从
# [0,255] 被缩放到了 [0,1]。
# 归一化为  [−1,1]：
# 使用均值 0.5 和标准差 0.5 的目的是将像素值进一步变换到
# [−1,1] 范围内，以便让模型更好地处理数据。

# 加载训练和测试数据集
train_dataset = datasets.FashionMNIST(
    root='data', train=True, transform=transform, download=True
)
test_dataset = datasets.FashionMNIST(
    root='data', train=False, transform=transform, download=True
)

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


def show_images(images, labels):
    # 创建一个 3x3 的图像网格
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title(f'Label: {labels[i]}')
        ax.axis('off')


# 获取一个批次的数据
data_iter = iter(train_loader)
images, labels = next(data_iter)
show_images(images, labels)
plt.show()


class Inception(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_pool):
        """
        参数：
        - in_channels: 输入的通道数
        - out_1x1: 1x1 卷积输出通道数
        - red_3x3: 3x3 卷积路径中 1x1 降维卷积的输出通道数
        - out_3x3: 3x3 卷积输出通道数
        - red_5x5: 5x5 卷积路径中 1x1 降维卷积的输出通道数
        - out_5x5: 5x5 卷积输出通道数
        - out_pool: 最大池化路径中的 1x1 卷积输出通道数
        """
        super(Inception, self).__init__()

        # 1x1 卷积分支
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)

        # 3x3 卷积分支，先降维再卷积
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, red_3x3, kernel_size=1),
            nn.Conv2d(red_3x3, out_3x3, kernel_size=3, padding=1)
        )

        # 5x5 卷积分支，先降维再卷积
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, red_5x5, kernel_size=1),
            nn.Conv2d(red_5x5, out_5x5, kernel_size=5, padding=2)
        )

        # 3x3 最大池化分支，接 1x1 卷积
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool, kernel_size=1)
        )

    def forward(self, x):
        # 每个分支的输出
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)

        # 将所有分支在通道维度上拼接
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)  # 按通道维度拼接

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogLeNet, self).__init__()

        # 初始卷积层
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)  # 输入为单通道灰度图
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 第一个 Inception 模块组
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 第二个 Inception 模块组
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 第三个 Inception 模块组
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        # 全局平均池化层
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接层
        self.fc = nn.Linear(1024, num_classes)  # 1024 是最后拼接后的通道数

    def forward(self, x):
        # 前向传播过程
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # 第一组 Inception 模块
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)

        # 第二组 Inception 模块
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.pool4(x)

        # 第三组 Inception 模块
        x = self.inception5a(x)
        x = self.inception5b(x)

        # 全局平均池化
        x = self.global_avg_pool(x)

        # 展平成向量
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.fc(x)
        return x


model = GoogLeNet(num_classes=10)  # 10 类分类
data_iter = iter(train_loader)
images, labels = next(data_iter)  # 获取图像和标签
print(f"输入图像的形状: {images.shape}")  # (batch_size, 1, 224, 224)

# 使用 GPU（如果可用）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # 将模型移动到 GPU 或 CPU
print(f"当前使用的设备: {device}")
model_device = next(model.parameters()).device
print(f"模型所在设备: {model_device}")

# 验证 GPU 上的计算是否正常
if torch.cuda.is_available():
        # 创建一个随机张量并移动到 GPU
    x = torch.randn(1, 1, 224, 224).to(device)
    print(f"测试张量所在设备: {x.device}")

        # 在模型上进行前向传播测试
    try:
        model.to(device)  # 将模型移动到 GPU
        output = model(x)  # 前向传播
        print("前向传播成功，模型正常使用 GPU。")
    except Exception as e:
        print(f"前向传播失败: {e}")
else:
    print("GPU 不可用，请检查 CUDA 环境或驱动程序。")
# 将数据移动到 GPU 或 CPU
images = images.to(device)
labels = labels.to(device)

# 前向传播，获取模型的输出
outputs = model(images)  # 前向传播
print(f"模型输出的形状: {outputs.shape}")  # (batch_size, 10)


# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用 Adam 优化器

# 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # 设置为训练模式
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

# 测试模型
model.eval()  # 设置为评估模式
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"测试集上的准确率: {100 * correct / total:.2f}%")
