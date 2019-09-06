# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 09:00:07 2019

@author: zhaozixiao
"""

import cifar10
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
FLAGS.data_dir = 'cifar10_data/'

cifar10.maybe_download_and_extract()