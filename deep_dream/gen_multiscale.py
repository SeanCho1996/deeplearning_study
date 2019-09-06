# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 15:49:11 2019

@author: zhaozixiao
"""

import tensorflow as tf
import numpy as np
import scipy.misc
import PIL.Image as pim

def savearray(img_array, img_name):
    pim.fromarray(np.uint8(img_array)).save(img_name)
    print('img saved: %s' % img_name)

def cal_grad_tiled(img, t_grad, tile_scale=512):
    sz = tile_scale
    h, w = img.shape[:2]
    
    # img_shift可以理解为fft_shift
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    
    for y in range(0, max(h-sz//2, sz), sz):
        for x in range(0, max(w-sz//2, sz), sz):
            sub = img_shift[y:y+sz, x:x+sz]
            g = sess.run(t_grad, {t_input: sub})
            grad[y:y+sz, x:x+sz] = g
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)


def resize_ratio(img, ratio):
    img_min = img.min()
    img_max = img.max()
    img = (img-img_min)/(img_max-img_min)*255 #把原图像的灰度值转换到0-255范围内做变换
    img = np.float32(scipy.misc.imresize(img, ratio))
    img = img/255*(img_max-img_min) + img_min # 回复原图的灰度值
    return img


def render_multiscale(t_obj, img0, iter_n=20, step=1.0, n=3, scale=1.5):
    t_score = tf.reduce_mean(t_obj)
    t_grad = tf.gradients(t_score, t_input)[0]
    
    img = img0.copy()
    for i in range(n):
        if i > 0:
            img = resize_ratio(img, scale)
        for it in range(iter_n):
            g = cal_grad_tiled(img, t_grad)
            g /= g.std() + 1e-8
            img += g*step
            print('.', end=' ')
    return img
            
    
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

name = 'mixed4d_3x3_bottleneck_pre_relu'
channel = 139
layer_output = graph.get_tensor_by_name("import/%s:0" % name)
img_noise = np.random.uniform(size=(224, 224, 3))+100.0
savearray(img_noise, 'img_ori.jpg')

img = render_multiscale(layer_output[:, :, :, channel], img_noise)
savearray(img, 'img_gene/img_c_x.jpg')