# -*- coding: utf-8 -*-
# @Author: bxsh
# @Email:  xbc0809@gmail.com
# @File:  test_SVM.py
# @Date:  2017/10/31 9:26


import os
import numpy as np
import cv2
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

path = './yale_face/'


def get_data(path, train=True, batch=100):
    file_names = sorted(os.listdir(path), key=lambda d: int(d[1:-4]))
    data = []
    label = []
    for i in range(0, len(file_names)):
        label.append(i // 11 + 1)
        data.append(cv2.imread(path + file_names[i], 0).flatten().tolist())
    return data, label


if __name__ == '__main__':
    X, Y = get_data(path)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    pca = PCA(n_components=100, svd_solver='randomized',
              whiten=True).fit(x_train)
    X_train_pca = pca.transform(x_train)
    X_test_pca = pca.transform(x_test)

    clf = svm.SVC(class_weight='balanced', max_iter=100)
    clf.fit(X_train_pca, y_train)
    pre = clf.predict(X_test_pca)
    print(pre)
    print(y_test)
    print((np.array(pre) == np.array(y_test)).sum() * 1.0 / len(pre))
    pass
