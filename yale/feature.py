# -*- coding: utf-8 -*-
# @Author: bxsh
# @Email:  xbc0809@gmail.com
# @File:  feature.py
# @Date:  2017/10/27 11:05


import os
import cv2
import numpy as np
import tensorflow as tf
from DL_Lib.dnn import conv2D
from skimage.feature import local_binary_pattern

path = './yale_face/s1.bmp'
if __name__ == '__main__':
    img = cv2.imread(path, 0)
    f = local_binary_pattern(img, 24, 3)
    cv2.imshow('1', f)
    cv2.waitKey()
