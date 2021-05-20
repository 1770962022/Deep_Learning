import torch 
from torchvision import datasets,transforms
import matplotlib.pyplot as plt 
from  torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

#导入模型文件model
from model import ConvNet

#模型训练次数
EPOCHS = 10
#数据批次的大小，即一批为BATCH_SIZE条数据
BATCH_SIZE = 512
#判断是否使用GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
#实例化一个网络，实例化后将网络送至DEVICE，gpu或者cpu
model =ConvNet().to(DEVICE)
#优化器选择Adom
optimizer =optim.Adam(model.parameters())

#训练模型
def train(model,device,train_loader,optimizer,epoch):
    #设置神经网络为训练
    model.train()
    #enumerate是遍历函数，可以遍历train_loader张量
    for batch_idx,(data,target) in enumerate(train_loader):
        #将数据送入设备，准备训练
        data,target = data.to(device),target.to(device)
        #梯度清零
        optimizer.zero_grad()
        #获得输入再模型里面的结果
        output = model(data)
        #计算loss_function
        loss = F.nll_loss(output,target)
        #反向传播
        loss.backward()
        #更新权重
        optimizer.step()
        #过程记录,epoch代表第几次训练，batch_idx*len(data)代表当前训练了的个数，len(train_loader)代表总数
        #100.*batch_idx/len(train_loader)代表百分制进度，loss.item()代表模型的准确程度
        if(batch_idx+1)%30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
#对模型进行测试
def test(model,device,test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # 将一批的损失相加
            test_loss += F.nll_loss(output, target, reduction='sum').item() 
            # 找到概率最大的下标
            # 因为输出是1*10的矩阵，分别代表输入是每个数字的概率，找出最大的就是输出结果
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
#开始训练
for epoch in range(1,EPOCHS+1):
    train(model,DEVICE,train_loader,optimizer,epoch)
    test(model,DEVICE,test_loader)

#保存模型
torch.save(model,'./model_1.pkl')




















