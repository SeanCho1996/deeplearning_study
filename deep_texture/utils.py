import vgg_16
from tensorflow import keras
import tensorflow as tf
from PIL import Image
import numpy as np
import time
import os


def pre_processing(input_image, height, width):
    img = Image.open(input_image)
    img = img.resize(size=(height, width))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array/255


def compute_vgg_output(input_image):
    start_time = time.time()
    net = vgg_16.VGG16()
    net.build(input_image)
    layer_list = dict({0: net.conv1_1, 1: net.conv1_2, 2: net.pool1,
                       3: net.conv2_1, 4: net.conv2_2, 5: net.pool2,
                       6: net.conv3_1, 7: net.conv3_2, 8: net.conv3_3, 9: net.pool3,
                       10: net.conv4_1, 11: net.conv4_2, 12: net.conv4_3, 13: net.pool4,
                       14: net.conv5_1, 15: net.conv5_2, 16: net.conv5_3, 17: net.pool5})

    output_list = dict()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(len(layer_list)):
            output_list[i] = keras.backend.eval(layer_list[i])
        print("all layers' outputs computed")
        end_time = time.time()
        print("character extraction finished, time used %f s" % (end_time - start_time))
    return output_list


def post_processing(conv_output, output_path, save_file=True):
    x = np.squeeze(conv_output)
    x = (x - np.amin(x))/(np.amax(x)-np.amin(x))
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    img = Image.fromarray(x, mode='RGB')
    img.show()

    if save_file:
        img.save(output_path)

    return x
