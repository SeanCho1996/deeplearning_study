# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 09:20:03 2019

@author: zhaozixiao
"""

import tensorflow as tf


with tf.Session() as sess:
    filename = ['a.jpg', 'b.jpg', 'c.jpg']
    filename_queue = tf.train.string_input_producer(filename, shuffle=False, num_epochs=1)
    
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    tf.local_variables_initializer().run()
    threads = tf.train.start_queue_runners(sess=sess)
    
    i = 0
    while True:
        i += 1
        image = sess.run(value)
        with open('read/test_%d.jpg' % i, 'wb') as f:
            f.write(image)
    
    