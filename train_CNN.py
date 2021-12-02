#!/usr/bin/env python
# encoding: utf-8

# @author: Zhipeng Ye
# @contact: Zhipeng.ye19@xjtlu.edu.cn
# @file: train_CNN.py
# @time: 2021-12-01 08:08
# @desc:
import processing as ps
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,      # input height
                out_channels=16,    # n_filters
                kernel_size=5,      # filter size
                stride=1,           # filter movement/step
                padding=2,      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
            ),      # output shape (16, 28, 28)
            nn.ReLU(),    # activation
            nn.MaxPool2d(kernel_size=2),    # 在 2x2 空间里向下采样, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)   # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output

if __name__ == '__main__':
    torch.manual_seed(1)  # reproducible

    # Hyper Parameters
    EPOCH = 5  # 训练整批数据多少次, 为了节约时间, 我们只训练一次
    BATCH_SIZE = 50
    LR = 0.001  # 学习率
    DOWNLOAD_MNIST = True  # 如果你已经下载好了mnist数据就写上 False

    # Mnist 手写数字
    train_data = torchvision.datasets.MNIST(
        root='./mnist/',  # 保存或者提取位置
        train=True,  # this is training data
        transform=torchvision.transforms.ToTensor(),  # 转换 PIL.Image or numpy.ndarray 成
        # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
        download=DOWNLOAD_MNIST,  # 没下载就下载, 下载了就不用再下了
    )

    test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)

    # 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    # 为了节约时间, 我们测试时只测试前2000个
    test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[
             :2000] / 255.  # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
    test_y = test_data.test_labels[:2000]

    cnn = CNN()

    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
    
    size = len(train_loader.dataset)
    # training and testing
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):  # 分配 batch data, normalize x when iterate train_loader
            output = cnn(b_x)  # cnn output
            loss = loss_func(output, b_y)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            if step % 100 == 0:
                loss, current = loss.item(), step * len(b_x)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


    PATH = 'cnn.pkl'
    torch.save(cnn.state_dict(), PATH)