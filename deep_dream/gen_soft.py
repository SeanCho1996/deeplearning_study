# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:24:51 2019

@author: zhaozixiao
"""

import numpy as np
import tensorflow as tf
import scipy
from functools import partial
import PIL.Image

k = np.float32([1, 4, 6, 4, 1])
k = np.outer(k, k)
k5x5 = k[:, :, None, None]/k.sum() * np.eye(3, dtype=np.float32)

graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)

# read in protobuff model, attention: here the model is just read in as data but not yet available as a tensor
model_fn = 'tensorflow_inception_graph.pb'
with tf.gfile.FastGFile(model_fn, mode='rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

t_input = tf.placeholder(tf.float32, name='t_input')
imagenet_mean = 117.0

# 把graph的input节点设为t_preprocessed张量
t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input': t_preprocessed})


def resize_ratio(img, ratio):
    img_min = img.min()
    img_max = img.max()
    img = (img-img_min)/(img_max-img_min)*255 #把原图像的灰度值转换到0-255范围内做变换
    img = np.float32(scipy.misc.imresize(img, ratio))
    img = img/255*(img_max-img_min) + img_min # 回复原图的灰度值
    return img


def cal_grad_tiled(img, t_grad, tile_scale=512):
    sz = tile_scale
    h, w = img.shape[:2]

    # img_shift可以理解为fft_shift
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)

    for y in range(0, max(h - sz // 2, sz), sz):
        for x in range(0, max(w - sz // 2, sz), sz):
            sub = img_shift[y:y + sz, x:x + sz]
            g = sess.run(t_grad, {t_input: sub})
            grad[y:y + sz, x:x + sz] = g
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)


def savearray(img_array, img_name):
    scipy.misc.toimage(img_array).save(img_name)
    print('img saved: %s' % img_name)


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
        std = tf.sqrt(tf.reduce_mean(tf.square(img)))  # 求标准差
    return img/tf.maximum(eps, std)


def lap_normalize(img, n=4):
    img = tf.expand_dims(img, 0)
    tlevels = lap_split_n(img, n)  # 利用刚刚完成的函数拉普拉斯分割原图
    tlevels = list(map(normalize, tlevels))  # 对每一层做标准化
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


# def render_lapnorm(t_obj, img0,
#                    iter_n=10, step=1.0, octave_n=3, octave_scale=1.4, lap_n=4):
#     t_score = tf.reduce_mean(t_obj)
#     t_grad = tf.gradients(t_score, t_input)[0]
#     # 将lap_norm转为正常函数？？？
#     lap_norm_func = tffunc(np.float32)(partial(lap_normalize,n=lap_n))
#
#     img = img0.copy()
#     for octave in range(octave_n):
#         if octave > 0:
#             img = resize_ratio(img, octave_scale)
#         for i in range(iter_n):
#             g = cal_grad_tiled(img, t_grad)
#             g = lap_norm_func(g)
#             img += g*step
#             print('.', end=' ')
#     savearray(img, 'lapnorm.jpg')


def render_deepdream(t_obj, img0,
                    iter_n=10, step=1.5, octave_n=4, octave_scale=1.4, lap_n=4):
    t_score = tf.reduce_mean(t_obj)
    t_grad = tf.gradients(t_score, t_input)[0]

    img = img0.copy()
    octaves = []
    for i in range(octave_n - 1):
        hw = img.shape[:2]
        lo = resize_ratio(img, np.int32(np.float32(hw)/octave_scale))
        hi = img - resize_ratio(lo, hw)
        img = lo
        octaves.append(hi)

    for octave in range(octave_n):
        if octave > 0:
            hi = octaves[-octave]
            img = resize_ratio(img, hi.shape[:2])+hi
        for i in range(iter_n):
            g =  cal_grad_tiled(img, t_grad)
            img += g*(step/(np.abs(g).mean() + 1e-7))
            print('.', end=' ')
    img = img.clip(0, 255)
    savearray(img, 'deep_dream.jpg')


if __name__ == '__main__':
    name = 'mixed4c'
    # channel = 139
    layer_output = graph.get_tensor_by_name("import/%s:0" % name)
    #img_noise = np.random.uniform(size=(224, 224, 3)) + 100.0
    img0 = PIL.Image.open('test.jpg')
    img0 = np.float32(img0)

    render_deepdream(layer_output, img0, iter_n=10)
