# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:24:51 2019

@author: zhaozixiao
"""

import numpy as np
import tensorflow as tf
import scipy

k = np.float32([1, 4, 6, 4, 1])
k = np.outer(k, k)
k5x5 = k[:, :, None, None] * np.eye(3, dtype=np.float32)


def lap_split(img):
    with tf.name_scope('split'):
        # 第一次高斯低通滤波，做平滑操作，取图像低频部分
        lo = tf.nn.conv2d(img, k5x5, [1, 2, 2, 1], 'SAME')  # size?????
        # 将滤波后的图片缩放到原图大小
        lo2 = tf.nn.conv2d_transpose(lo, k5x5*4, tf.shape(img), [1, 2, 2, 1])  # k5x5 * 4 ??????
        hi = img - lo2
    return lo, hi


def lap_split_n(img, n):
    levels = []
    for i in range(n):
        img, hi = lap_split(img)  # 原图做n次低频分割
        levels.append(hi)  # 保存每次低频分割出的高频部分
    levels.append(img)  # 最后保存最低频部分，形成金字塔
    return levels[::-1]


def lap_merge(levels):
    img = levels[0]
    with tf.name_scope('merge'):
        for hi in levels[1:]:
            img = tf.nn.conv2d_transpose(img, k5x5*4, tf.shape(hi), [1, 2, 2,1]) + hi
    return img


def normalize(img, eps=1e-10):
    with tf.name_scope('normalize'):
        std = tf.sqrt(tf.reduce_mean(tf.square(img)))
    return img/tf.maximum(eps, std)


def lap_normalize(img, n=4):
    img = tf.expand_dims(img, 0)
    tlevels = lap_split_n(img, n)
    tlevels = list(map(normalize, tlevels))
    out = lap_merge(tlevels)
    return out[0, :, :, :]


# https://foofish.net/python-decorator.html查看关于wrap的讲解
def tffunc(*argtypes):
    placeholders = list(map(tf.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, ** kwargs):
            return out.eval(dict(zip(placeholders, args)), session=kwargs.get('session'))
        return wrapper
    return wrap