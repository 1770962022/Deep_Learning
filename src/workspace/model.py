import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__( self ):
        super().__init__()
        #每次会送入BATCH_SIZE*1*28*28个样本，输入通道为1，图像采用的是灰度图，像素28*28
        #Conv2d是卷积层，第一个参数是输入通道数，第二个参数是输出通道数，第三个是卷积核,第四个是步长，第五个是填充
        #模型采用两次卷积，两次池化
        #输入通道是1，输出通道是16，卷积核的大小是5*5,步长是1，padding是2   
        self.conv1 = nn.Conv2d(1,16,5,1,2)
        #输入通道是16，输出通道是32，卷积核大小是5*5，步长是1，padding是2
        self.conv2 = nn.Conv2d(16,32,5,1,2)
        #Linear是全连接层，第一个参数是输入通道数，第二个是输出通道数
        #经过最后一次池化后，Linear的输入通道数变为7*7*32，输出通道数是128
        self.fc1 = nn.Linear(7*7*32,128)
        #输入通道数是128，输出通道是10，即预测0-9，十个数
        self.fc2 = nn.Linear(128,10)
    def forward(self,x):
        #在本次实验中in_size是512，即BATCH_SIZE的值，输入的x可以看成是512*1*28*28的张量
        in_size = x.size(0)
        #经过第一次的卷积，由1*28*28——>16*28*28，经过一次填充，再经过一次5*5的卷积，输出不变还是28*28
        out = self.conv1(x)
        #使用ReLU激活函数，不改变形状
        out = F.relu(out)
        #第一次池化层，经过步长是2，池化f是2，向量的行列都d减半，由16*28*28——>16*14*14
        out = F.max_pool2d(out,2,2)
        #第二次卷积，卷积核，步长，以及padding和第一次都是一样的,只是输出通道变了。16*14*14——>32*14*14
        out = self.conv2(out)
        #使用ReLU函数激活
        out = F.relu(out)
        #再经过一次f=2,S=2的池化层。32*14*14——>32*7*7
        out = F.max_pool2d(out,2,2)
        #转换到全连接层,使用view方法进行数据转化，参数第一个是BATCH_SIZE的值，第二个是-1 代表自动推算
        out = out.view(in_size,-1)
        #由全连接层7*7*32——>128
        out = self.fc1(out)
        #还是使用ReLU激活函数
        out = F.relu(out)
        #得到预测结果，即128——>10
        out = self.fc2(out)
        #计算log(softmax(x)),归一化指数函数
        out = F.log_softmax(out,dim=1)
        return out










        
        
    
