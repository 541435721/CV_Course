# -*- coding: utf-8 -*-
# @Author: bxsh
# @Email:  xbc0809@gmail.com
# @File:  read_data.py
# @Date:  2017/11/15 14:54


import cv2
import numpy as np
import os


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


if __name__ == '__main__':
    train = unpickle('./cifar-100-python/train')
    # test = unpickle('./cifar-100-python/test')
    labels_name = unpickle('./cifar-100-python/meta')
    imgs = train[b'data']
    labels = train[b'fine_labels']
    names = labels_name[b'fine_label_names']
    data_counter = dict(zip(names, [0] * len(names)))
    PATH = './img/'
    for i in range(3, imgs.shape[0]):
        img = imgs[i, :].reshape(3, 32, 32)  # R G B
        img = img.transpose([1, 2, 0])
        # img[..., 0], img[..., 1], img[..., 2] = img[..., 1], img[..., 2], img[..., 0]
        img = cv2.resize(img, (128, 128))
        cv2.imshow('test', img)
        cv2.waitKey()
        # cat = str(names[labels[i]], encoding='utf-8')
        # CLASS = os.listdir(PATH)
        # if cat not in CLASS:
        #     os.mkdir(PATH + cat)
        # cv2.imwrite(PATH + cat + '/' + str(data_counter[names[labels[i]]]) + '.jpg', img)
        # data_counter[names[labels[i]]] += 1
