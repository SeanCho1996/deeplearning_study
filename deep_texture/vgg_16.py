import numpy as np
import tensorflow as tf
from tensorflow import keras
import time


vgg_mean = [103.939, 116.779, 123.68]

#  here we reconstruct a vgg16 network without FC layer and replacing max-pooling with avg-pooling

class VGG16:
    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = 'vgg16.npy'
        self.vgg_param = np.load(path, encoding='latin1', allow_pickle=True).item()

        print('npy file loaded')
        # print(self.vgg_param)

    def get_filter_param(self, name):
        return keras.backend.constant(self.vgg_param[name][0], name="filter")

    def get_bias(self, name):
        return keras.backend.constant(self.vgg_param[name][1], name="bias")

    def conv_layer(self, input, name):
        # read parameters from vgg16.npy
        filter = self.get_filter_param(name)
        bias = self.get_bias(name)

        conv = keras.backend.conv2d(input, filter, strides=(1, 1), padding="same")
        with_bias = keras.backend.bias_add(conv, bias)
        output = keras.backend.relu(with_bias)

        return output

    def avg_pooling(self, input, name):
        return keras.backend.pool2d(input, pool_size=(2, 2), strides=(2, 2), pool_mode='avg')

    def build(self, rgb_image):
        r, g, b = tf.split(rgb_image, num_or_size_splits=3, axis=3)
        bgr_image = tf.concat(values=[b - vgg_mean[0]/255, g - vgg_mean[1]/255, r - vgg_mean[2]/255], axis=3)

        start_time = time.time()

        self.conv1_1 = self.conv_layer(bgr_image, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.avg_pooling(self.conv1_2, "pool1")

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.avg_pooling(self.conv2_2, "pool2")

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.avg_pooling(self.conv3_3, "pool3")

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.avg_pooling(self.conv4_3, "pool4")

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.avg_pooling(self.conv5_3, "pool5")