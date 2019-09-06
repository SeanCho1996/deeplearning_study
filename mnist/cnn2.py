# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 09:38:11 2019

@author: zhaozixiao
"""


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time


def gaussien_filter(shape):
    filtre = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(filtre)


def bias(shape):
    bias = tf.constant(0.1, shape=shape)
    return tf.Variable(bias)


def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def pooling_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

x_image = tf.reshape(x, [-1, 28, 28, 1])

# cnn 1
kernel1 = gaussien_filter([5, 5, 1, 32])
bias1 = bias([32])
cons1 = tf.nn.relu(conv2d(x_image, kernel1)+bias1)
pool1 = pooling_2x2(cons1)

# cnn 2
kernel2 = gaussien_filter([5, 5, 32, 64])
bias2 = bias([64])
cons2 = tf.nn.relu(conv2d(pool1, kernel2)+bias2)
pool2 = pooling_2x2(cons2)

# fcn
fcn_kernel = gaussien_filter([7*7*64, 1024])
fcn_bias = bias([1024])
fcn_flat = tf.reshape(pool2, [-1, 7*7*64])
fcn_cons1 = tf.nn.relu(tf.matmul(fcn_flat, fcn_kernel)+fcn_bias)

dropout = tf.placeholder(tf.float32)
fcn_drop = tf.nn.dropout(fcn_cons1, dropout)

fcl_kernel = gaussien_filter([1024, 10])
fcl_bias = ([10])
fcl_flat = tf.matmul(fcn_cons1, fcl_kernel)+fcl_bias

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=fcl_flat))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.arg_max(y_, 1), tf.arg_max(fcl_flat, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

starttime = time.time()
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], dropout: 1.0})
        print("step %d, training accuracy: %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], dropout: 0.5})
endtime = time.time()
print("train time: ", endtime-starttime)

print("test accuracy: %g" % (accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, dropout: 1.0})))
