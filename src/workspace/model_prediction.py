import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
from matplotlib import pyplot as plt
from model import ConvNet
#装载数据
BATCH_SIZE = 512
train_file = datasets.MNIST(
    root='./dataset/',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)
test_file = datasets.MNIST(
    root='./dataset/',
    train=False,
    transform=transforms.ToTensor()
)
train_loader = DataLoader(
    dataset=train_file,
    batch_size=BATCH_SIZE,
    shuffle=True
)
test_loader = DataLoader(
    dataset=test_file,
    batch_size=BATCH_SIZE,
    shuffle=False
)
DEVICE = torch.device("cpu")
#使用已经训练好的模型
model_name = './model_1.pkl'
model = torch.load(model_name)
# 模型为测试状态
model.eval()
#从测试集中选取测试数据
for data, targets in test_loader:
    data = data[:9]
    targets = targets[:9]
    break
output = model(data).argmax(1)
plt.figure(figsize=(9, 9))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.title(f'pred {output[i]} | true {targets[i]}')
    plt.axis('off')
    plt.imshow(data[i].squeeze(0), cmap='gray')
plt.savefig('./prediction.jpg')
