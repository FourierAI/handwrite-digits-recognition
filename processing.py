#!/usr/bin/env python
# encoding: utf-8

# @author: Zhipeng Ye
# @contact: Zhipeng.ye19@xjtlu.edu.cn
# @file: processing.py
# @time: 2021-11-29 15:01
# @desc:

from sklearn import datasets
import numpy as np
import struct
import matplotlib.pyplot as plt


def load_dataset():
    digits = datasets.load_digits()
    labels = digits.target
    images = digits.images
    return images, labels


def load_MNIST_train_dataset():
    with open('dataset/train-images-idx3-ubyte', 'rb') as file:
        magic, num, rows, cols = struct.unpack('>IIII', file.read(16))
        images = np.fromstring(file.read(), dtype=np.uint8).reshape(num, 28, 28)

    with open('dataset/train-labels-idx1-ubyte', 'rb') as file:
        magic, n = struct.unpack('>II', file.read(8))
        labels = np.fromstring(file.read(), dtype=np.uint8)

    return images, labels


def load_MNIST_test_dataset():
    with open('dataset/t10k-images-idx3-ubyte', 'rb') as file:
        magic, num, rows, cols = struct.unpack('>IIII', file.read(16))
        images = np.fromstring(file.read(), dtype=np.uint8).reshape(num, 28, 28)

    with open('dataset/t10k-labels-idx1-ubyte', 'rb') as file:
        magic, n = struct.unpack('>II', file.read(8))
        labels = np.fromstring(file.read(), dtype=np.uint8)

    return images, labels


def scaling(images):
    MAX_VALUE = 255
    max_value = np.max(images)

    scale = MAX_VALUE / max_value

    images = images * scale.astype(np.uint8)

    return images


def reshaping(images):
    samples, height, width = images.shape
    images = np.reshape(images, (samples, height * width))
    return images


def binarization(images):
    images[images > 0] = 255
    return images


def centering(image):
    row_num, col_num = image.shape
    rows, cols = np.where(image > 0)
    if len(rows) == 0 or len(cols) == 0:
        return image

    min_row = np.min(rows)
    max_row = np.max(rows)
    min_col = np.min(cols)
    max_col = np.max(cols)

    # padding_row = row_num - (max_row-min_row+1)
    # padding_col = col_num - (max_col-min_col+1)
    centering_row = (min_row + max_row + 1) // 2
    centering_col = (min_col + max_col + 1) // 2

    raw_center_row = row_num // 2
    raw_center_col = col_num // 2

    diff_row = raw_center_row - centering_row
    diff_col = raw_center_col - centering_col

    new_min_row = min_row + diff_row
    new_max_row = max_row + diff_row
    new_min_col = min_col + diff_col
    new_max_col = max_col + diff_col

    new_image = np.zeros(shape=(row_num, col_num))

    new_image[new_min_row:new_max_row + 1, new_min_col:new_max_col + 1] = image[min_row:max_row + 1,
                                                                          min_col:max_col + 1]
    return new_image


if __name__ == "__main__":
    image = np.zeros(shape=(8, 8))
    # image[5:8, 6] = 255
    # image[0:6, 3:8] = 255
    plt.imshow(image)
    plt.show()
    image = centering(image)

    plt.imshow(image)
    plt.show()
