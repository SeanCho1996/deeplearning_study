# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 09:51:43 2019

@author: zhaozixiao
"""

import tensorflow as tf
import os
import cifar10_input
import scipy

def inputs_origin(data_dir):
    filename =[os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)]
    for f in filename:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to read file: ' + f)
    filename_queue = tf.train.string_input_producer(filename)
    data_input = cifar10_input.read_cifar10(filename_queue)
    
    reshaped_image = tf.cast(data_input.uint8image, tf.float32)
    
    return reshaped_image
    
    
with tf.Session() as sess:
    reshaped_image = inputs_origin('cifar10_data/cifar-10-batches-bin/')
    threads = tf.train.start_queue_runners(sess=sess)
    tf.local_variables_initializer().run()
    if not os.path.exists('cifar10_data/raw/'):
        os.makedirs('cifar10_data/raw/')
    for i in range(30):
        image_array = sess.run(reshaped_image)
        scipy.misc.toimage(image_array).save('cifar10_data/raw/test_%d.jpg' % i)
        