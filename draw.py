#!/usr/bin/env python
# encoding: utf-8

# @author: Zhipeng Ye
# @contact: Zhipeng.ye19@xjtlu.edu.cn
# @file: draw.py
# @time: 2021-11-29 08:23
# @desc:

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import processing as ps

if __name__ == '__main__':

    # digits = datasets.load_digits()
    # labels = digits.target
    # images = digits.images

    # 直接查看图片
    # plt.imshow(images[0])
    # plt.show()

    # 灰度图像显示
    # plt.imshow(images[0], cmap='gray')
    # plt.show()
    
    # 子图显示图片
    # plt.figure(1)
    # for i in range(9):
    #     plt.subplot(3, 3, i+1)
    #     plt.imshow(images[i], cmap = 'gray')
    # plt.show()

    # 显示MNIST数据集
    images, labels = ps.load_MNIST_train_dataset()
    plt.imshow(images[1])
    plt.show()
