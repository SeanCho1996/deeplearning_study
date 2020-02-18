import utils
from tensorflow import keras
import numpy as np


# file path
input_img = 'original/texture_2.jpg'
output_img = 'processed/texture_2_processed.png'

# output size
height = 128
width = 128


def loss_function(num_layers, feature_map, noise_feature_map):
    '''
    :param num_layers: int, number of convolution layers
    :param feature_map: vgg16 outputs for each convolutional layer of the original image
    :param noise_feature_map: vgg16 outputs for each concolutional layer of the noise image
    :return: difference between the grim matrix of original/noise image
    '''
    total_loss = keras.backend.constant(0, dtype=np.float, name='loss')
    for i in range(len(num_layers)):
        texture_feature = np.squeeze(feature_map[[1][0]], 0)
        texture_feature = np.reshape(texture_feature, texture_feature.shape[0] * texture_feature.shape[1], texture_feature.shape[2])
        grim_texture = np.matmul(texture_feature.T, texture_feature)

        noise_feature = np.squeeze(noise_feature_map[[1][0]], 0)
        noise_feature = np.reshape(noise_feature.shape[0] * noise_feature.shape[1], noise_feature.shape[2])
        grim_noise = keras.matmul(noise_feature.T, noise_feature)




if __name__ == '__main__':
    img_array = utils.pre_processing(input_img, height, width)
    outputs = utils.compute_vgg_output(img_array)