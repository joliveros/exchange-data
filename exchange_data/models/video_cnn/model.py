#!/usr/bin/env python

import alog
import click
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv3D, MaxPooling3D, ZeroPadding3D, \
    Flatten, Dense, Dropout, GlobalAveragePooling3D
from tensorflow.keras.models import Sequential

# Activation = tf.keras.layers.Activation
# BatchNormalization = tf.keras.layers.BatchNormalization
# Conv1D = tf.keras.layers.Conv1D
# Conv2D = tf.keras.layers.Conv2D
# ConvLSTM2D = tf.keras.layers.ConvLSTM2D
# Dense = tf.keras.layers.Dense
# Dropout = tf.keras.layers.Dropout
# Flatten = tf.keras.layers.Flatten
# GlobalAveragePooling1D = tf.keras.layers.GlobalAveragePooling1D
# GlobalAveragePooling2D = tf.keras.layers.GlobalAveragePooling2D
# Input = tf.keras.Input
# LeakyReLU = tf.keras.layers.LeakyReLU
# LSTM = tf.keras.layers.LSTM
# MaxPooling2D = tf.keras.layers.MaxPooling2D
# Reshape = tf.keras.layers.Reshape
# Sequential = tf.keras.models.Sequential
# TimeDistributed = tf.keras.layers.TimeDistributed


def Model(
    depth,
    sequence_length,
    include_last=False,
    input_shape=None,
    learning_rate=5e-5,
    num_categories=2,
    **kwargs
):
    if not input_shape:
        input_shape = (6, sequence_length, depth * 2, 1)

    inputs = Input(shape=input_shape)

    alog.info(inputs)

    conv = C3D(input_shape, **kwargs).model(inputs)

    dense_out = Dense(
        num_categories, activation='softmax',
        # use_bias=True,
        # bias_initializer=tf.keras.initializers.Constant(value=[0.0, 1.0])
    )

    out = dense_out(conv)

    if out.shape.as_list() != [None, 2] and include_last:
        raise Exception()

    model = tf.keras.Model(inputs=inputs, outputs=out)

    if include_last:
        model.compile(
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
            optimizer=tf.keras.optimizers.Adadelta(learning_rate=learning_rate,
                                                   clipnorm=1.0)
        )
    return model


class C3D:
    def __init__(
        self,
        input_shape,
        base_filter_size=64,
        dense_size=64,
        kernel_size=3,
        strides=2,
        **kwargs
    ):
        self.kernel_size = kernel_size
        self.strides = strides
        self.conv_count = 0
        self.pool_count = 0
        model = Sequential()

        # 1st layer group
        model.add(self.conv(base_filter_size, input_shape=input_shape))
        # model.add(self.max_pooling(pool_size=(1, 2, 2)))

        # 2nd layer group
        model.add(self.conv(base_filter_size * 2))
        # model.add(self.max_pooling())

        # 3rd layer group
        model.add(self.conv(base_filter_size * 4))
        model.add(self.conv(base_filter_size * 4))
        # model.add(self.max_pooling())

        # 4th layer group
        # model.add(ZeroPadding3D(padding=(0, 1, 1), input_shape=input_shape))
        model.add(self.conv(base_filter_size * 8))
        model.add(self.conv(base_filter_size * 8))
        # model.add(self.max_pooling())

        # model.add(GlobalAveragePooling3D())

        model.add(Flatten())

        # FC layers group
        model.add(Dense(dense_size, activation='relu', name='fc6'))
        # model.add(Dropout(.5))
        # model.add(Dense(dense_size, activation='relu', name='fc7'))
        model.add(Dropout(.5))
        # model.add(Dense(int(dense_size / 2), activation='softmax', name='fc8'))

        for layer in model.layers:
            layer.trainable = True

        self.model = model
        self.model.summary()

    def max_pooling(self, pool_size=(2, 2, 2), **kwargs):
        layer = MaxPooling3D(
            pool_size=pool_size,
            strides=(1, 2, 2),
            name=f'pool_{self.pool_count}',
            padding="valid", **kwargs)

        self.pool_count += 1
        return layer

    def conv(self, base_filter_size, **kwargs):
        kernel_size = self.kernel_size
        strides = self.strides

        conv = Conv3D(
            base_filter_size, (kernel_size, kernel_size, kernel_size),
            activation="relu",
            name=f'conv_{self.conv_count}',
            padding="same",
            strides=(strides, strides, strides),
            **kwargs
        )

        self.conv_count += 1
        return conv


@click.command()
@click.option('--batch-size', '-b', type=int, default=1)
@click.option('--depth', type=int, default=32)
@click.option('--sequence-length', type=int, default=48)
def main(**kwargs):
    Model(**kwargs)


if __name__ == '__main__':
    main()
