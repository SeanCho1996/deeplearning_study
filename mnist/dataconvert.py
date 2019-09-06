# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 14:46:22 2019

@author: zhaozixiao
"""

from tensorflow.examples.tutorials.mnist import input_data
import scipy.misc
import os

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

save_dir = "MNIST_data/raw/"
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)
    
for i in range(5000):
    image_array = mnist.train.images[i, :]
    image_array = image_array.reshape(28, 28)
    file_name = save_dir + '%d.jpg' % i
    scipy.misc.toimage(image_array, cmin=0.0, cmax=1.0).save(file_name)
    
save_dir = "MNIST_data/validation/"
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)
 
for i in range(500):
    image_array = mnist.validation.images[i, :]
    image_array = image_array.reshape(28, 28)
    file_name = save_dir + '%d.jpg' % i
    scipy.misc.toimage(image_array, cmin=0.0, cmax=1.0).save(file_name)