import torch.nn as nn #定义自定义网络继承nn.Module
import torch

class Net(nn.Module):
    def __init__(self, c):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential( # conv卷积层，ReLU为激活函数，MaxPool2d为池化层
            nn.Conv2d(3, 32, 3, 1, 1),  # 输入通道rgb3通道，输出通道32个特征图，卷积核3x3，步长1，填充1
            nn.ReLU(),
            nn.MaxPool2d(2)#2x2的池化
        )  # 32x14x14
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),  #32通道到64通道，3x3卷积核，步长1，填充1
            nn.ReLU(),
            nn.MaxPool2d(2)  #减半输出 64x7x7
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),  # 输入64x7x7
            nn.ReLU(),
            nn.MaxPool2d(2)  # 输出64x3x3
        )
        self.dense = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),  # fc4 64*3*3 -> 128
            nn.ReLU(),
            nn.Linear(128, c)  # fc5 128->10
        )
    def forward(self, x): # 进行三次卷积
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)  # 64x3x3
        res = conv3_out.view(conv3_out.size(0), -1)  # batch x (64*3*3)
        #res = torch.reshape(conv3_out, (conv3_out.size(0), 64*3*3))
        out = self.dense(res) #全连接层，输出结果
        return out
