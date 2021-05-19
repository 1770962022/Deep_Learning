import torch 
from torchvision import datasets,transforms
import matplotlib.pyplot as plt 
from  torch.utils.data import DataLoader

#模型训练次数
EPOCH = 10 
#数据批次的大小，即一批为BATCH_SIZE条数据
BATCH_SIZE = 512

#下载数据集
train_file = datasets.MNIST(
        root = "./dataset",
        train = True,
        transform = transforms.ToTensor(),
        download = True
        )

test_file = datasets.MNIST(
        root = "./dataset",
        train = False,
        transform = transforms.ToTensor(),
        )   
#取测试集数据
test_data = test_file.data
test_targets = test_file.targets
#加载数据集
train_loader = DataLoader(
        dataset = train_file,
        batch_size = BATCH_SIZE,
        shuffle = True
        )

test_loader = DataLoader(
        dataset = test_file,
        batch_size = BATCH_SIZE,
        shuffle = False
        )



plt.figure(figsize=(9,9))
for i in range(9):
    plt.subplot (3,3,i+1)
    plt.imshow(test_data[i],cmap='gray')
    plt.axis('off')
plt.savefig('./ts.jpg')

        

