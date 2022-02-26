#!/usr/bin/env python

import alog
import click
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv3D, MaxPooling3D, ZeroPadding3D, \
    Flatten, Dense, Dropout, GlobalAveragePooling3D, LeakyReLU
from tensorflow.keras.models import Sequential

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
        input_shape = (4, sequence_length, depth * 2, 1)

    inputs = Input(shape=input_shape)

    alog.info(inputs)

    conv = C3D(input_shape, **kwargs).model(inputs)

    dense_out = Dense(
        num_categories, activation='softmax',
        use_bias=False,
        # bias_initializer=tf.keras.initializers.Constant(value=[0.0, 0.99])
    )

    out = dense_out(conv)

    if out.shape.as_list() != [None, 2] and include_last:
        raise Exception()

    model = tf.keras.Model(inputs=inputs, outputs=out)

    if include_last:
        model.compile(
            loss='mse',
            metrics=['accuracy'],
            optimizer=tf.keras.optimizers.Adadelta(learning_rate=learning_rate,
                                                   clipnorm=1.0)
        )
    return model


class C3D:
    def __init__(
        self,
        input_shape,
        base_filter_size=32,
        dense_size=256,
        kernel_size=1,
        strides=1,
        **kwargs
    ):
        self.kernel_size = kernel_size
        self.strides = strides

        if kernel_size > 4:
            self.strides = 3

        self.conv_count = 0
        self.pool_count = 0
        model = Sequential()

        # 1st layer group
        # model.add(ZeroPadding3D(padding=(0, 1, 1), input_shape=input_shape))
        model.add(self.conv(base_filter_size, input_shape=input_shape))
        model.add(self.max_pooling())

        # 2nd layer group
        model.add(self.conv(base_filter_size * 2))
        model.add(self.conv(base_filter_size * 2))
        model.add(self.max_pooling())

        # 3rd layer group
        model.add(self.conv(base_filter_size * 4))
        model.add(self.conv(base_filter_size * 4))
        model.add(self.max_pooling())

        # 4th layer group
        # model.add(ZeroPadding3D(padding=(0, 1, 1), input_shape=input_shape))
        model.add(self.conv(base_filter_size * 8))
        model.add(self.conv(base_filter_size * 8))
        model.add(self.max_pooling())

        # 4th layer group
        model.add(ZeroPadding3D(padding=(0, 1, 1), input_shape=input_shape))
        model.add(self.conv(base_filter_size * 8))
        model.add(self.conv(base_filter_size * 8))
        model.add(self.max_pooling())

        model.add(GlobalAveragePooling3D())

        # model.add(Flatten())

        # FC layers group
        model.add(Dense(dense_size, activation='relu', name='fc6'))
        model.add(Dropout(.1))
        # model.add(Dense(dense_size, activation='relu', name='fc7'))
        # model.add(Dropout(.1))
        # model.add(Dense(dense_size, activation='relu', name='fc8'))
        # model.add(Dropout(.1))
        # model.add(Dense(dense_size, activation='relu', name='fc9'))

        for layer in model.layers:
            layer.trainable = True

        self.model = model
        self.model.summary()

    def max_pooling(self, pool_size=(1, 2, 2), **kwargs):
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
            activation=LeakyReLU(),
            name=f'conv_{self.conv_count}',
            padding="same",
            strides=(self.strides, self.strides, self.strides),
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
