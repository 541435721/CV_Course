# -*- coding: utf-8 -*-
# @Author: bxsh
# @Email:  xbc0809@gmail.com
# @File:  dsn_net.py
# @Date:  2017/10/9 14:14


import tensorflow as tf
import numpy as np
from DL_Lib.dnn import conv3D, deconv3D

DSN = [
    ['block1', [
        ['conv', [9, 9, 7, 1, 8], [1, 1, 1, 1, 1], 'SAME', 0.7],
    ]],
    ['block2', [
        ['conv', [7, 7, 5, 8, 16], [1, 1, 1, 1, 1], 'SAME', 0.7],
        ['pool', [1, 2, 2, 2, 1], [1, 2, 2, 2, 1], 'SAME'],
    ]],
    ['block3', [
        ['conv', [7, 7, 5, 16, 32], [1, 1, 1, 1, 1], 'SAME', 0.7],
    ]],
    ['block4', [
        ['conv', [5, 5, 3, 32, 32], [1, 1, 1, 1, 1], 'SAME', 0.7],
        ['pool', [1, 2, 2, 2, 1], [1, 2, 2, 2, 1], 'SAME'],
    ]],
    ['block5', [
        ['conv', [1, 1, 1, 32, 32], [1, 1, 1, 1, 1], 'SAME', 0.7],
    ]],
    ['block6', [
        ['conv', [1, 1, 1, 32, 32], [1, 1, 1, 1, 1], 'SAME', 0.7],
    ]],
    ['block7', [
        ['deconv', [3, 3, 3, 32, 32], [1, 2, 2, 2, 1], 'SAME'],
    ]],
    ['block8', [
        ['deconv', [3, 3, 3, 2, 32], [1, 2, 2, 2, 1], 'SAME'],
    ]],
]

supervise = [
    ['supervise_1', 3, [
        ['deconv', [3, 3, 3, 2, 16], [1, 2, 2, 2, 1], 'SAME'],
    ]],
    ['supervise_2', 6, [
        ['deconv1', [3, 3, 3, 32, 32], [1, 2, 2, 2, 1], 'SAME'],
        ['deconv2', [3, 3, 3, 2, 32], [1, 2, 2, 2, 1], 'SAME'],
    ]],
]

g = tf.Graph()

layers = []

with g.as_default():
    with tf.name_scope('input'):
        X = tf.placeholder(dtype=tf.float32, shape=[None, None, None, None, 1])  # [batch,batchsize,w,h,c]
        Y = tf.placeholder(dtype=tf.int64, shape=[None, None, None, None])  # [batch,batchsize,w,h]
        W = tf.placeholder(dtype=tf.float32, shape=[4])
    layers.append(X)
    for block in DSN:
        with tf.variable_scope(block[0]):
            for layer in block[1]:
                with tf.variable_scope(layer[0]):
                    if layer[0].startswith('conv'):
                        conv = conv3D(layer[0], layers[-1], layer[1], layer[2], layer[3], layer[4])
                        layers.append(conv)
                    elif layer[0].startswith('pool'):
                        pool = tf.nn.max_pool3d(layers[-1], layer[1], layer[2], layer[3])
                        layers.append(pool)
                    elif layer[0].startswith('deconv'):
                        deconv = deconv3D(layer[0], layers[-1], layer[1], layer[2], layer[3])
                        layers.append(deconv)
    out = [layers[-1]]

    for block in supervise:
        with tf.variable_scope(block[0]):
            temp = layers[block[1]]
            for layer in block[2]:
                with tf.variable_scope(layer[0]):
                    temp = deconv3D(layer[0], temp, layer[1], layer[2], layer[3])
            out.append(temp)

    for i in range(len(out)):
        out[i] = tf.nn.softmax(out[i])
        # out[i] = tf.argmax(temp, axis=-1)

if __name__ == '__main__':
    with tf.Session(graph=g) as sess:
        saver = tf.train.Saver()
        tf.global_variables_initializer().run()
        x = np.zeros([1, 128, 128, 128, 1])
        ans = sess.run(out, feed_dict={X: x})
        for i in ans:
            print(i.shape)
            # summary_writer = tf.summary.FileWriter('./summary', graph=sess.graph)
            # saver.save(sess, './test_model_save/test.ckpt')
