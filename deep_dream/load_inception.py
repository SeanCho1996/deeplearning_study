# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 09:36:19 2019

@author: zhaozixiao
"""

from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.misc


def savearray(img_array, img_name):
    scipy.misc.toimage(img_array).save(img_name)
    print('img saved: %s' % img_name)    
    
    
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

layers = [op.name for op in graph.get_operations() if op.type == 'Conv2D' and 'import/' in op.name]
# print('Number of layers', len(layers))
name = 'mixed4d_3x3_bottleneck_pre_relu'
channel = 120
layer_output = graph.get_tensor_by_name("import/%s:0" % name)
img_noise = np.random.uniform(size=(224,224,3))+100.0

def render_naive(t_obj, img0, iter_n=20, step=1.0):
    t_score = tf.reduce_mean(t_obj)
    t_grad = tf.gradients(t_score, t_input)[0]
    
    img = img0.copy()
    for i in range(iter_n):
        g, score = sess.run([t_grad, t_score], {t_input: img})
        g /= g.std() + 1e-8
        img += g*step
        print('score(mean)=%f' % (score))
    savearray(img, 'naive.jpg')
    
    
render_naive(layer_output[:, :, :, channel], img_noise, iter_n=20)



    
