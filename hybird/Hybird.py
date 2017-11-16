# -*- coding: utf-8 -*-
# @Author: bxsh
# @Email:  xbc0809@gmail.com
# @File:  Hybird.py
# @Date:  2017/11/15 18:17

import tensorflow as tf
import numpy as np
import os

BATCH_SIZE = 2
generator_arch = [
    ['conv1', [5, 5, 3, 6], [1, 1, 1, 1], 'SAME', 0.75, True, True, 0.003],
    ['conv2', [5, 5, 9, 9], [1, 1, 1, 1], 'SAME', 0.75, True, True, 0.003],
    ['conv3', [5, 5, 18, 18], [1, 1, 1, 1], 'SAME', 0.75, True, True, 0.003],
    ['conv4', [5, 5, 36, 36], [1, 1, 1, 1], 'SAME', 0.75, True, True, 0.003],
    ['conv5', [5, 5, 72, 72], [1, 1, 1, 1], 'SAME', 0.75, True, True, 0.003],
]
generator_layers = []

discriminator_arch = [
    ['conv1', [5, 5, 3, 6], [1, 1, 1, 1], 'VALID', 0.75, True, True, 0.003],
    ['pool1', [1, 2, 2, 1], [1, 2, 2, 1], 'SAME'],
    ['conv2', [5, 5, 6, 16], [1, 1, 1, 1], 'VALID', 0.75, True, True, 0.003],
    ['pool2', [1, 2, 2, 1], [1, 2, 2, 1], 'SAME'],
    ['fc1', 1024],
    ['fc2', 512],
    ['fc3', 2]
]
discriminator_layers = []


def FC(in_layer, name, out_nodes):
    shape = in_layer.get_shape().as_list()
    nodes = shape[1]
    for i in range(2, len(shape)):
        nodes *= shape[i]
    reshaped = tf.reshape(in_layer, [shape[0], nodes])
    W = tf.get_variable(name + '_W', shape=[nodes, out_nodes], dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
    b = tf.get_variable(name + '_b', shape=[out_nodes], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

    out = tf.matmul(reshaped, W) + b
    return out


def conv2D(in_layer, name, kernel_size, stride, padding='SAME', Dropout=None, BN=False, Activate=False,
           regular=None):
    kernel = tf.get_variable(name=name + '_W', shape=kernel_size, dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(0.1, stddev=0.01))
    b = tf.get_variable(name=name + '_b', shape=[kernel_size[-1]], dtype=tf.float32,
                        initializer=tf.constant_initializer(0.1))

    with tf.variable_scope('conv2d'):
        out = tf.nn.conv2d(in_layer, kernel, stride, padding)
        out = tf.nn.bias_add(out, b)
    if Dropout:
        with tf.variable_scope('Dropout'):
            out = tf.nn.dropout(out, Dropout)
    if BN:
        with tf.variable_scope('BN'):
            out = tf.layers.batch_normalization(out, training=True)
    if Activate:
        with tf.variable_scope('ACTIVATE'):
            out = tf.nn.relu(out)
    if regular:
        l2_loss = tf.contrib.layers.l2_regularizer(regular)(kernel)
        tf.add_to_collection('regular', l2_loss)
    return out


def deconv2D(in_layer, name, kernel_size, stride, padding='SAME'):
    in_shape = tf.shape(in_layer)
    kernel = tf.get_variable(name + '_W', shape=kernel_size, dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(0.0, 0.01))
    b = tf.get_variable(name + '_b', shape=[kernel_size[-2]], dtype=tf.float32,
                        initializer=tf.constant_initializer(0.1))
    output_shape = tf.stack(
        [in_shape[0], in_shape[1] * 2, in_shape[2] * 2, kernel_size[-2]])
    with tf.variable_scope('deconv2d'):
        out = tf.nn.conv2d_transpose(in_layer, kernel, output_shape, strides=stride, padding=padding)
        out = tf.nn.relu(tf.nn.bias_add(out, b))
    return out


def generator(X):
    generator_layers.append(X)
    with tf.variable_scope('generator'):
        for layer in generator_arch:
            with tf.variable_scope(layer[0]):
                if layer[0].startswith('conv'):
                    temp = generator_layers[0]
                    for i in range(1, len(generator_layers)):
                        temp = tf.concat([temp, generator_layers[i]], axis=-1)
                    out = conv2D(temp, layer[0], layer[1], layer[2], layer[3],
                                 layer[4], layer[5],
                                 layer[6], layer[7])
                    generator_layers.append(out)
        with tf.variable_scope('output'):
            output = conv2D(generator_layers[-1], 'output', [1, 1, 72, 3], [1, 1, 1, 1], Activate=True)
        return output


def discriminator(X, reuse=False):
    discriminator_layers.append(X)
    with tf.variable_scope('discriminator', reuse=reuse):
        for layer in discriminator_arch:
            with tf.variable_scope(layer[0], reuse=reuse):
                if layer[0].startswith('conv'):
                    out = conv2D(discriminator_layers[-1], layer[0], layer[1], layer[2], layer[3], layer[4], layer[5],
                                 layer[6], layer[7])
                    discriminator_layers.append(out)
                if layer[0].startswith('pool'):
                    out = tf.nn.max_pool(discriminator_layers[-1], layer[1], layer[2], layer[3])
                    discriminator_layers.append(out)
                if layer[0].startswith('fc'):
                    out = FC(discriminator_layers[-1], layer[0], layer[1])
                    discriminator_layers.append(out)
        output = tf.nn.softmax(discriminator_layers[-1])
    return output


g = tf.Graph()
with g.as_default():
    with tf.variable_scope('input'):
        X = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 256, 256, 3])
        Y = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 256, 256, 3])
    output = generator(X)
    pre = discriminator(output)
    pre2 = discriminator(Y, reuse=True)

if __name__ == '__main__':
    with tf.Session(graph=g) as sess:
        Writer = tf.summary.FileWriter('./summary', graph=sess.graph)
        x = np.zeros([2, 256, 256, 3])
        tf.global_variables_initializer().run()
        ans = sess.run(tf.shape(pre), feed_dict={X: x})
        print(ans)
    pass
