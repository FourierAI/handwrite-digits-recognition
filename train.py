#!/usr/bin/env python
# encoding: utf-8

# @author: Zhipeng Ye
# @contact: Zhipeng.ye19@xjtlu.edu.cn
# @file: train.py
# @time: 2021-11-29 17:44
# @desc:
import processing as ps
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.externals import joblib

if __name__ == '__main__':
    np.random.seed(6)

    # 读取sklearn digits characters
    # images, labels = ps.load_dataset()

    # 读取MNIST数据集
    images, labels = ps.load_MNIST_train_dataset()

    images = ps.scaling(images)
    images = ps.reshaping(images)

    # X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3)
    X_train, y_train = images, labels

    # KNN算法
    # model = KNeighborsClassifier()
    # params = {"n_neighbors": [i for i in range(5,10)]}

    # 逻辑回归算法
    # model = LogisticRegression()
    # params = {"C": [i*0.1 for i in range(1, 50)]}

    #多层感知机
    model = MLPClassifier()
    # SVM算法
    # model = svm.SVC('linear')
    # params = {"C": [i * 0.1 for i in range(1, 5)]}

    # model = GridSearchCV(model, params, cv=5)
    model.fit(X_train, y_train)
    # print(f'model.best_params_:{model.best_params_}')
    joblib.dump(model, 'mlp.pkl')

    images, labels = ps.load_MNIST_test_dataset()
    images = ps.scaling(images)
    images = ps.reshaping(images)

    X_test, y_test = images, labels

    accuracy = model.score(X_test, y_test)
    print(f'accuracy:{accuracy}')

    y_pred = model.predict(X_test)
    con_mat = confusion_matrix(y_test, y_pred)
    print(con_mat)

