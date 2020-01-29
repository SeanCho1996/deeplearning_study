import tensorflow as tf
from tensorflow import keras
import __future__


def conv3x3(input, output_channels, strides=1, padding='same'):
    x = keras.layers.Conv2D(kernel_size=(3, 3), filters=output_channels, strides=strides, padding=padding)(input)
    return x


def conv1x1(input, output_channels, strides=1, padding='same'):
    x = keras.layers.Conv2D(kernel_size=(1, 1), filters= output_channels, strides=strides, padding=padding)(input)
    return x


def res_block_2(input, input_filter, output_filter):
    # 3*3*16 conv
    x = conv3x3(input, output_filter)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    # 3*3*16 conv
    x = conv3x3(x, output_filter)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    # if input size and convolution output size does not match, need to do downsampling
    if input_filter != output_filter:
        print("input, output sizes don't match, downsampling required")
        input = conv1x1(input, output_filter)

    x = keras.layers.add([x, input])
    x = keras.layers.Activation('relu')(x)

    return x


def res_block_3(input, input_filter, output_filter):
    # 3*3*32 conv
    x = conv3x3(input, output_filter)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    # 3*3*32 conv
    x = conv3x3(x, output_filter)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    if input_filter != output_filter:
        print("input, output sizes don't match, downsampling required")
        input = conv1x1(input, output_filter)

    x = keras.layers.add([x, input])
    x = keras.layers.Activation('relu')(x)

    return x


def build_model(input_size):
    # define input layer
    input = keras.layers.Input(shape=input_size)

    # conv1
    x = conv3x3(input, 16)

    # res_block 2
    x = res_block_2(x, 16, 16)
    x = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2, 2))(x)

    # res_block 3
    x = res_block_3(x, 16, 32)
    x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    # avg_pool
    x = keras.layers.AvgPool2D(pool_size=(7, 7))(x)

    # flatten
    x = keras.layers.Flatten()(x)

    # fully connected
    x = keras.layers.Dense(10, activation='softmax')(x)

    # make model
    model = keras.Model(inputs=input, outputs=x)
    keras.utils.plot_model(model, to_file='resnet_mnist.png')
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    return model


def train(model, x_train, x_test, y_train, y_test):
    check_point = keras.callbacks.ModelCheckpoint('./model',
                                                  monitor='var_loss',
                                                  verbose=1,
                                                  save_best_only=True,
                                                  save_weights_only=False,
                                                  mode='auto',
                                                  period=5)
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
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    if keras.backend.image_data_format == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
        x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
        input_shape = (1, 28, 28)
    else:
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        input_shape = (28, 28, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    _model = build_model(input_shape)

    train(_model, x_train, x_test, y_train, y_test)

if __name__=='__main__':
    tf.app.run()
