# -*- coding: utf-8 -*-
# @Author: bxsh
# @Email:  xbc0809@gmail.com
# @File:  LeNet.py
# @Date:  2017/9/28 10:30


import os
import cv2
import numpy as np
import tensorflow as tf
from DL_Lib.dnn import conv2D
from skimage.feature import local_binary_pattern

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

path = '/home/bxsh/face/yale_face/'

OUTPUT_NODE = 15
Data_size = 10 * 8

IMAGE_SIZE = 100
NUM_CHANNELS = 1
NUM_LABELS = 15

CONV1_DEEP = 32
CONV1_SIZE = 5

CONV2_DEEP = 64
CONV2_SIZE = 3

FC_SIZE_1 = 512
FC_SIZE_2 = 256
BATCH_SIZE = 50

LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.9
TRAIN_STEP = 5000

g = tf.Graph()

with g.as_default():
    X = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1], name='input')
    Y_ = tf.placeholder(dtype=tf.float32, shape=[None, NUM_LABELS])

    conv1 = tf.nn.conv2d(X, tf.get_variable(name='conv1', shape=[CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                            dtype=tf.float32,
                                            initializer=tf.truncated_normal_initializer(0.1, stddev=0.01)),
                         strides=[1, 1, 1, 1],
                         padding="SAME")
    bias1 = tf.nn.relu(tf.nn.bias_add(conv1, tf.get_variable('bias1', shape=[CONV1_DEEP], dtype=tf.float32,
                                                             initializer=tf.constant_initializer(0.1))))
    pool1 = tf.nn.max_pool(bias1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    conv2 = tf.nn.conv2d(pool1, tf.get_variable(name='conv2', shape=[CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                                dtype=tf.float32,
                                                initializer=tf.truncated_normal_initializer(0.0, stddev=0.01)),
                         strides=[1, 1, 1, 1],
                         padding="SAME")
    bias2 = tf.nn.relu(tf.nn.bias_add(conv2, tf.get_variable('bias2', shape=[CONV2_DEEP], dtype=tf.float32,
                                                             initializer=tf.constant_initializer(0.0))))
    pool2 = tf.nn.max_pool(bias2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    fcn1 = tf.matmul(reshaped, tf.get_variable('fcn_weight1', shape=[nodes, FC_SIZE_1], dtype=tf.float32,
                                               initializer=tf.truncated_normal_initializer(0.0,
                                                                                           stddev=0.01))) + tf.get_variable(
        'fcn_bias1', shape=[FC_SIZE_1], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
    relu1 = tf.nn.relu(fcn1)

    fcn2 = tf.matmul(relu1, tf.get_variable('fcn_weight2', shape=[FC_SIZE_1, FC_SIZE_2], dtype=tf.float32,
                                            initializer=tf.truncated_normal_initializer(0.0,
                                                                                        stddev=0.01))) + tf.get_variable(
        'fcn_bias2', shape=[FC_SIZE_2], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
    relu2 = tf.nn.relu(fcn2)

    softmax = tf.matmul(relu2, tf.get_variable('soft_max_weight', shape=[FC_SIZE_2, OUTPUT_NODE], dtype=tf.float32,
                                               initializer=tf.truncated_normal_initializer(0.0,
                                                                                           stddev=0.01))) + tf.get_variable(
        'softmax_bias', shape=[OUTPUT_NODE], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=softmax, labels=tf.argmax(Y_, 1))
    cross_entropy_means = tf.reduce_mean(cross_entropy)

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, Data_size / BATCH_SIZE,
                                               LEARNING_RATE_DECAY, staircase=True)
    train = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_means, global_step)

    correct_prediction = tf.equal(tf.argmax(softmax, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('loss', cross_entropy_means)
    tf.summary.scalar('acc', accuracy)
    merged = tf.summary.merge_all()


def get_data(path, train=True, batch=BATCH_SIZE):
    file_names = sorted(os.listdir(path), key=lambda d: int(d[1:-4]))
    data = []
    label = []
    for i in range(0, len(file_names)):
        temp_label = [0] * 15
        temp_label[i // 11] = 1
        label.append(temp_label)
        if i % 11 == 0:
            data.append(file_names[i:i + 11])
    label = np.array(label).reshape([15, 11, 15])
    data = np.array(data)
    # [15,11]
    if train:
        x = np.random.randint(0, 15, batch)
        y = np.random.randint(0, 9, batch)
    else:
        x = np.random.randint(0, 15, batch)
        y = np.random.randint(9, 11, batch)

    selected_name = []
    selected_label = []
    selected_data = np.zeros([batch, IMAGE_SIZE, IMAGE_SIZE])
    count = 0
    for i in range(len(x)):
        selected_name.append(data[x[i], y[i]])
        selected_label.append(label[x[i], y[i]].tolist())
        img = cv2.imread('./yale_face/' + data[x[i], y[i]], 0)
        selected_data[count, :, :] = img
        count += 1

    selected_label = np.array(selected_label)

    selected_data = selected_data[..., np.newaxis]
    return selected_data, selected_label


if __name__ == '__main__':
    with tf.Session(graph=g) as sess:
        saver = tf.train.Saver()
        summary_writer1 = tf.summary.FileWriter('./summary/train', graph=sess.graph)
        summary_writer2 = tf.summary.FileWriter('./summary/test', graph=sess.graph)
        tf.global_variables_initializer().run()
        count = 0
        for i in range(TRAIN_STEP):
            xs, ys = get_data(path)
            xt, yt = get_data(path, False)
            # print(xs.shape, ys.shape)
            # print(xt.shape, yt.shape)
            # x = xs.reshape(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
            _, acc1, l1, l2, loss, m = sess.run(
                [train, accuracy, tf.argmax(softmax, 1), tf.argmax(ys, 1), cross_entropy_means, merged],
                {X: xs, Y_: ys})
            acc2, mt = sess.run([accuracy, merged], {X: xt, Y_: yt})
            if i % 10 == 0:
                print(acc1, loss)
                print(l1)
                print(l2)
                summary_writer1.add_summary(m, count)
                summary_writer2.add_summary(mt, count)
                count += 1
