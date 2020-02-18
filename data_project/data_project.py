import tensorflow as tf
from tensorflow import keras
import pandas
import random
import numpy as np
from sklearn.model_selection import train_test_split


def conv1x3(inpt, output_channels, strides=1, padding='same'):
    x = keras.layers.Conv1D(kernel_size=(3), filters=output_channels, strides=strides, padding=padding)(inpt)
    return x


def conv1x1(inpt, output_channels, strides=1, padding='same'):
  x = keras.layers.Conv1D(kernel_size=1, filters=output_channels, strides=strides, padding=padding)(inpt)
  return x


def res_block_2(inpt, input_filter, output_filter=16):
    # 3*16 conv
    x = conv1x3(inpt, output_filter)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    # 3*16 conv
    x = conv1x3(x, output_filter, strides=1)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    # if input size and convolution output size does not match, need to do downsampling
    if input_filter != output_filter:
        print("input, output sizes don't match, downsampling required")
        inpt = conv1x1(inpt, output_filter)

    x = keras.layers.add([x, inpt])
    x = keras.layers.Activation('relu')(x)

    return x


def res_block_3(inpt, input_filter, output_filter=32):
    # 3*32 conv
    x = conv1x3(inpt, output_filter)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    # 3*32 conv
    x = conv1x3(x, output_filter, strides=1)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    if input_filter != output_filter:
        print("input, output sizes don't match, downsampling required")
        inpt = conv1x1(inpt, output_filter)

    x = keras.layers.add([x, inpt])
    x = keras.layers.Activation('relu')(x)

    return x


def build_model(input_size):
    # define input layer
    inpt = keras.layers.Input(shape=input_size)

    print(inpt.shape)
    # conv1
    x = conv1x3(inpt, 16)

    # res_block 2
    x = res_block_2(x, 16, 16)
    # x = keras.layers.MaxPool1D(pool_size=(2,2), strides=(2, 2))(x)

    # res_block 3
    x = res_block_3(x, 16, 32)
    # x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    # flatten
    x = keras.layers.Flatten()(x)

    # fully connected
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dense(8, activation='softmax')(x)

    # make model
    model = keras.Model(inputs=inpt, outputs=x)
    keras.utils.plot_model(model, to_file='resnet_test.png')
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    return model


def train_jb(model, x_train, x_test, y_train, y_test):
    print('fgsfgsfgsg')
    check_point = keras.callbacks.ModelCheckpoint('./model',
                                                  monitor='var_loss',
                                                  verbose=1,
                                                  save_best_only=True,
                                                  save_weights_only=False,
                                                  mode='auto',
                                                  period=5)
    print('fhskjfgjq')
    model.fit(x_train, y_train,
              batch_size=50,
              epochs = 100,
              verbose=2,
              validation_data=(x_test, y_test),
              callbacks=[check_point])


def test(model, x_test, y_test):
    model.load_wights('./model')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    score = model.evaluate(x_test, y_test, verbose=0)
    print('test loss: ', score[0])
    print('test accuracy: ', score[1])


def main(_):
      '''
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        if keras.backend.image_data_format == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
            x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
            input_shape = (1, 28, 28)
        else:
            x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
            x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
            input_shape = (28, 28, 1)
      '''
      scaled_data = pandas.read_csv('./data.csv')
      train,test=train_test_split(scaled_data,test_size=0.15,random_state=0,stratify=scaled_data['Response'])
      x_train = np.array(train[train.columns[train.columns!='Response']])
      x_train = x_train.reshape(x_train.shape[0], 97, 1)
      y_train = np.array(train[train.columns[-5]].values - 1)
      x_test = np.array(test[test.columns[test.columns!='Response']])
      x_test = x_test.reshape(x_test.shape[0], 97, 1)
      y_test = np.array(test[test.columns[-5]].values -1)
      # Y=scaled_data['Response']
      # X=scaled_data[scaled_data.columns[scaled_data.columns!='Response']]
      # scaled_data.to_csv('./data.csv')

      x_train = x_train.astype('float32')
      x_test = x_test.astype('float32')

      batch_input_shape = (97, 1)

      # x_train /= 255
      # x_test /= 255

      y_train = keras.utils.to_categorical(y_train, 8)
      y_test = keras.utils.to_categorical(y_test, 8)

      _model = build_model(batch_input_shape)


      train_jb(_model, x_train, x_test, y_train, y_test)
      # print('qdqgsqfgshsrtghjds')


if __name__ == '__main__':
    tf.app.run()