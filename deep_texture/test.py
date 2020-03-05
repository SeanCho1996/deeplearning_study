import utils
from tensorflow import keras
import tensorflow as tf
import numpy as np
import vgg_16
from matplotlib import pyplot as plt


# file path
input_img = 'original/texture_2.jpg'
output_img = 'processed/texture_2_processed.png'

# output size
height = 128
width = 128


def loss_function(weights, feature_map, noise_feature_map):
    '''
    :param weights: weights for each convolutional layer's output feature map
    :param feature_map: vgg16 outputs for each convolutional layer of the original image
    :param noise_feature_map: vgg16 outputs for each convolutional layer of the noise image
    :return: difference between the grim matrix of original/noise image
    '''
    total_loss = keras.backend.constant(0, dtype=tf.float32, name='loss')  # tf.constant(0, dtype=tf.float32, name='loss')
    for i in range(len(weights)):
        texture_feature = np.squeeze(feature_map[[i][0]], 0)
        texture_feature = np.reshape(texture_feature, newshape=(texture_feature.shape[0] * texture_feature.shape[1], texture_feature.shape[2]))
        gram_texture = np.matmul(texture_feature.T, texture_feature)  # we construct the gram matrix with the auto-correlation for each feature map

        noise_feature = keras.backend.squeeze(noise_feature_map[[i][0]], 0)  # tf.squeeze(noise_feature_map[[i][0]], axis=0)  #
        noise_feature = keras.backend.reshape(noise_feature, shape=(noise_feature.shape[0] * noise_feature.shape[1], noise_feature.shape[2]))
        gram_noise = keras.backend.dot(keras.backend.transpose(noise_feature), noise_feature)

        denominator = (4 * keras.backend.constant(texture_feature.shape[0], dtype=tf.float32)**2) * keras.backend.constant(texture_feature.shape[1], dtype=tf.float32)**2

        total_loss += weights[i][0] * (keras.backend.sum(keras.backend.square(tf.subtract(gram_texture, gram_noise))) / keras.backend.cast(denominator, tf.float32))

    return total_loss



if __name__ == '__main__':
    # generate original feature maps
    img_array = utils.pre_processing(input_img, height, width)
    feature_map = utils.compute_vgg_output(img_array)

    # generate initial noise image
    random_ = keras.backend.random_uniform(img_array.shape, minval=0, maxval=0.2)
    noise_img = keras.backend.variable(value=random_, dtype=tf.float32, name="noise_input")

    # compute feature maps of initial noise map
    vgg = vgg_16.VGG16()
    vgg.build(noise_img)

    noise_layers_list = dict({0: vgg.conv1_1, 1: vgg.conv1_2, 2: vgg.pool1,
                              3: vgg.conv2_1, 4: vgg.conv2_2, 5: vgg.pool2,
                              6: vgg.conv3_1, 7: vgg.conv3_2, 8: vgg.conv3_3, 9: vgg.pool3,
                              10: vgg.conv4_1, 11: vgg.conv4_2, 12: vgg.conv4_3, 13: vgg.pool4,
                              14: vgg.conv5_1, 15: vgg.conv5_2, 16: vgg.conv5_3, 17: vgg.pool5})

    # we define the same weight for each layer
    m = [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1),(6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1), (12, 1),
         (13, 1)]
    # m = [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1), (12, 1),
    #     (13, 1), (14, 1), (15, 1), (16, 1), (17, 1)]

    loss = loss_function(m, feature_map, noise_layers_list)
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    epochs = 10000

    # init_image = keras.backend.eval(noise_img)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        init_image = sess.run(noise_img)
        for i in range(epochs):
            _, s_loss = sess.run([optimizer, loss])
            if (i+1)%100 == 0:
                print("Epoch:{} / {}".format(i+1, epochs), "Loss:", s_loss)
        final_noise = sess.run(noise_img)

    init_noise = utils.post_processing(init_image, output_img, save_file=False)
    final_image = utils.post_processing(final_noise, output_img, save_file=True)