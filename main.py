#encoding=utf-8
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import cv2 as cv

def loadtraindata():
    path = r"training"                                         # 路径
    trainset = torchvision.datasets.ImageFolder(path,
                                                transform=transforms.Compose([
                                                    transforms.Resize((32, 32)),  # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小

                                                    transforms.CenterCrop(32),
                                                    transforms.ToTensor()])
                                                )

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True)
    return trainloader


class Net(nn.Module):  # 定义网络，继承torch.nn.Module
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 卷积层
        self.pool = nn.MaxPool2d(2, 2)  # 池化层
        self.conv2 = nn.Conv2d(6, 16, 5)  # 卷积层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 全连接层
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)  # 10个输出


    def forward(self, x):  # 前向传播

        x = self.pool(F.relu(self.conv1(x)))  # F就是torch.nn.functional

        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  # .view( )是一个tensor的方法，使得tensor改变size但是元素的总数是不变的。
        # 从卷基层到全连接层的维度转换

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

classes = ('hongjun','yiduo')

def loadtestdata():
    path = r"testing"
    testset = torchvision.datasets.ImageFolder(path,
                                                transform=transforms.Compose([
                                                    transforms.Resize((32, 32)),  # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小
                                                    transforms.ToTensor()])
                                                )
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=True)
    return testloader


def trainandsave():
    trainloader = loadtraindata()
    # 神经网络结构
    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # 学习率为0.001
    criterion = nn.CrossEntropyLoss()  # 损失函数也可以自己定义，我们这里用的交叉熵损失函数
    # 训练部分

    for epoch in range(100):  # 训练的数据量为5个epoch，每个epoch为一个循环
        # 每个epoch要训练所有的图片，每训练完成200张便打印一下训练的效果（loss值）
        running_loss = 0.0  # 定义一个变量方便我们对loss进行输出
        for i, data in enumerate(trainloader, 0):  # 这里我们遇到了第一步中出现的trailoader，代码传入数据
            # enumerate是python的内置函数，既获得索引也获得数据
            # get the inputs
            inputs, labels = data  # data是从enumerate返回的data，包含数据和标签信息，分别赋值给inputs和labels

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)  # 转换数据格式用Variable

            optimizer.zero_grad()  # 梯度置零，因为反向传播过程中梯度会累加上一次循环的梯度

            # forward + backward + optimize
            outputs = net(inputs)  # 把数据输进CNN网络net
            loss = criterion(outputs, labels)  # 计算损失值
            loss.backward()  # loss反向传播
            optimizer.step()  # 反向传播后参数更新
            running_loss += loss.item()  # loss累加
            if i % 20 == 19:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))  # 然后再除以200，就得到这两百次的平均损失值
                running_loss = 0.0  # 这一个200次结束后，就把running_loss归零，下一个200次继续使用

    print('Finished Training')
    # 保存神经网络
    torch.save(net, 'net.pkl')  # 保存整个神经网络的结构和模型参数
    torch.save(net.state_dict(), 'net_params.pkl')  # 只保存神经网络的模型参数


#trainandsave()

def reload_net():
    trainednet = torch.load('net.pkl')
    return trainednet

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def test():
    testloader = loadtestdata()
    net = reload_net()
    dataiter = iter(testloader)
    images, labels = dataiter.next()  #
    img = cv.imread('0.jpg')
    img2 = cv.resize(img, (32,32))
    print(img.shape)  # numpy数组格式为（H,W,C）
    transf = transforms.ToTensor()
    img_tensor = transf(img2)  # tensor数据格式是torch(C,H,W)
    print(img_tensor)
    #print('GroundTruth: '
    #      , " ".join('%5s' % classes[labels[0]]))  # 打印前25个GT（test集里图片的标签）
    outputs = net(Variable(torch.unsqueeze(img_tensor, dim=0).float()))
    _, predicted = torch.max(outputs.data, 1)
    print(outputs.data)
    print('Predicted: ', " ".join('%5s' % classes[predicted[0]]))
    # 打印预测值
    #imshow(torchvision.utils.make_grid(img_tensor, nrow=1))  # nrow是每行显示的图片数量，缺省值为8
    cv.imshow("img", img2)
    cv.imshow('imshow', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

test()
