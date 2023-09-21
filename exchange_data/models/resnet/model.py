#!/usr/bin/env python
import inspect
from copy import copy

import alog
import click
import tensorflow as tf

Activation = tf.keras.layers.Activation
BatchNormalization = tf.keras.layers.BatchNormalization
Conv1D = tf.keras.layers.Conv1D
Conv2D = tf.keras.layers.Conv2D
ConvLSTM2D = tf.keras.layers.ConvLSTM2D
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Flatten = tf.keras.layers.Flatten
GlobalAveragePooling1D = tf.keras.layers.GlobalAveragePooling1D
GlobalAveragePooling2D = tf.keras.layers.GlobalAveragePooling2D
Input = tf.keras.Input
LeakyReLU = tf.keras.layers.LeakyReLU
LSTM = tf.keras.layers.LSTM
MaxPooling2D = tf.keras.layers.MaxPooling2D
Reshape = tf.keras.layers.Reshape
Sequential = tf.keras.models.Sequential
TimeDistributed = tf.keras.layers.TimeDistributed


def Model(
    batch_size=16,
    lstm_size=16,
    num_lstm=0,
    num_dense=0,
    dense_width=32,
    include_last=False,
    input_shape=None,
    learning_rate=5e-5,
    num_categories=2,
    **kwargs
):
    # alog.info(alog.pformat((dense_width, kwargs)))
    # tf.compat.v1.experimental.output_all_intermediates(True)

    inputs = Input(shape=input_shape)

    conv = ResNetTS(
        input_shape,
        **kwargs
    ).model(inputs)


    dense_out = Dense(
       num_categories,
       activation='softmax',
       use_bias=True,
       bias_initializer=tf.keras.initializers.Constant(value=[0.6, 0.5])
    )

    dense_out.trainable = True

    out = dense_out(conv)

    # if include_last:
    #     out = dense_out(dense)
    # else:
    #     out = dense

    if out.shape.as_list() != [None, 2] and include_last:
        raise Exception()

    model = tf.keras.Model(inputs=inputs, outputs=out)

    if include_last:
        model.compile(
            # loss='sparse_categorical_crossentropy',
            # metrics=['accuracy'],
            # optimizer=tf.keras.optimizers.Adadelta(learning_rate=learning_rate,
            #                                        clipnorm=1.0)
        )
    return model


class ResNetTS:
    def __init__(
    self,
        input_shape,
        conv_block_strides=2,
        max_pooling_enabled=False,
        gap_enabled=True,
        base_filter_size=32,
        block_kernel=2,
        kernel_size=2,
        max_pooling_kernel=2,
        max_pooling_strides=2,
        num_categories=2,
        num_conv=16,
        padding=1,
        strides=1,
        **kwargs
    ):
        input = Input(input_shape)

        self.bn_axis = 3
        self.conv_block_strides = conv_block_strides

        conv = tf.keras.layers.ZeroPadding2D(padding=(padding, padding))(input)
        conv = Conv2D(filters=base_filter_size,
                      kernel_size=(kernel_size, kernel_size),
                      strides=(strides, strides),
                      padding='valid', kernel_initializer='he_normal')(conv)

        conv = tf.keras.layers.BatchNormalization(axis=self.bn_axis)(conv)
        conv = Activation('relu')(conv)

        conv = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv)
        if max_pooling_enabled:
            conv = tf.keras.layers.MaxPooling2D(
                (max_pooling_kernel, max_pooling_kernel),
                strides=(max_pooling_strides, max_pooling_strides))(conv)

        filters = [base_filter_size, base_filter_size, base_filter_size]
        kernel_size = (block_kernel, block_kernel)

        filters2 = [filter * 2 for filter in filters]
        filters3 = [filter * 4 for filter in filters]
        filters4 = [filter * 8 for filter in filters]

        layer_keys = [(int(key.split('conv_layer_')[1]), key)
                      for key in kwargs.keys() if 'conv_layer_' in key]

        def _filters(conv_ix, interval=4):
            scale = int(conv_ix / interval) + 1

            f = [filter * (2 ** scale) for filter in filters]
            return f

        conv = self.conv_block(conv, kernel_size, _filters(0), [1, 1])

        if len(layer_keys) > 0:
            for ix, key in layer_keys:
                try:
                    if kwargs[key] == 'conv':
                        conv = self.conv_block(conv, kernel_size, _filters(ix))
                    elif kwargs[key] == 'identity':
                        conv = self.identity_block(conv, kernel_size,
                                                   _filters(ix))
                        conv = self.identity_block(conv, kernel_size,
                                                   _filters(ix))
                        conv = self.identity_block(conv, kernel_size,
                                                   _filters(ix))
                except ValueError:
                    conv = self.conv_block(conv, kernel_size, _filters(ix))
        else:
            convs = [
                lambda conv: self.conv_block(conv, kernel_size, filters, [1, 1]),
                lambda conv: self.identity_block(conv, kernel_size, filters),
                lambda conv: self.identity_block(conv, kernel_size, filters),
                lambda conv: self.conv_block(conv, kernel_size, filters2),
                lambda conv: self.identity_block(conv, kernel_size, filters2),
                lambda conv: self.identity_block(conv, kernel_size, filters2),
                lambda conv: self.conv_block(conv, kernel_size, filters2),
                lambda conv: self.identity_block(conv, kernel_size, filters2),
                lambda conv: self.identity_block(conv, kernel_size, filters2),
                lambda conv: self.conv_block(conv, kernel_size, filters2),
                lambda conv: self.identity_block(conv, kernel_size, filters3),
                lambda conv: self.identity_block(conv, kernel_size, filters3),
                lambda conv: self.conv_block(conv, kernel_size, filters3),
                lambda conv: self.identity_block(conv, kernel_size, filters3),
                lambda conv: self.identity_block(conv, kernel_size, filters3),
                lambda conv: self.conv_block(conv, kernel_size, filters3),
                lambda conv: self.identity_block(conv, kernel_size, filters3),
                lambda conv: self.identity_block(conv, kernel_size, filters3),
                lambda conv: self.conv_block(conv, kernel_size, filters3)
            ]

            if num_conv > 0 and len(layer_keys) == 0:
                convs = convs[:num_conv]

            last_identity_block = None
            for _conv in convs:
                if 'conv_block' in inspect.getsource(_conv):
                    last_identity_block = _conv
                try:
                    conv = _conv(conv)
                except ValueError:
                    conv = last_identity_block(conv)

                # alog.info(conv.shape)
            
        output = GlobalAveragePooling2D()(conv)
 

        #if gap_enabled:
        #    output = GlobalAveragePooling2D()(conv)
        #else:
        #    output = Flatten()(conv)

        # alog.info(output.shape)
        self.model = tf.keras.models.Model(inputs=input, outputs=output)

        for layer in self.model.layers:
            layer.trainable = True

        self.model.summary()

    def conv_block(self, input_tensor, kernel_size, filters, strides=None):
        if strides is None:
            strides = self.conv_block_strides

        conv = Conv2D(filters=filters[0], kernel_size=[1, 1], strides=strides,
                      kernel_initializer='he_normal', padding='valid')(
            input_tensor)
        conv = tf.keras.layers.BatchNormalization(axis=self.bn_axis)(conv)
        conv = Activation('relu')(conv)

        conv = Conv2D(filters=filters[1], kernel_size=kernel_size,
                      padding='same',
                      kernel_initializer='he_normal')(conv)
        conv = tf.keras.layers.BatchNormalization(axis=self.bn_axis)(conv)
        conv = Activation('relu')(conv)

        conv = Conv2D(filters=filters[2], padding='valid', kernel_size=[1, 1],
                      kernel_initializer='he_normal')(conv)

        conv = tf.keras.layers.BatchNormalization(axis=self.bn_axis)(conv)

        short_cut = Conv2D(filters=filters[2], kernel_size=[1, 1],
                           strides=strides,
                           padding='valid', kernel_initializer='he_normal')(
            input_tensor)

        short_cut = tf.keras.layers.BatchNormalization(axis=self.bn_axis)(
            short_cut)

        output = tf.keras.layers.add([conv, short_cut])
        output = Activation('relu')(output)

        return output

    def identity_block(self, input_tensor, kernel_size, filters):
        filters1, filters2, filters3 = filters

        x = Conv2D(filters1, (1, 1),
                   kernel_initializer='he_normal', padding='valid')(
            input_tensor)
        x = BatchNormalization(axis=self.bn_axis)(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size, (1, 1),
                   padding='same',
                   kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=self.bn_axis)(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1),
                   kernel_initializer='he_normal', padding='valid')(x)
        x = BatchNormalization(axis=self.bn_axis)(x)

        x = tf.keras.layers.add([x, input_tensor])
        x = Activation('relu')(x)
        return x


@click.command()
@click.option('--batch-size', '-b', type=int, default=1)
@click.option('--depth', type=int, default=40)
@click.option('--sequence-length', type=int, default=48)
def main(**kwargs):
    kwargs['conv_layer_0'] = 'conv'
    kwargs['conv_layer_1'] = 'identity'
    kwargs['conv_layer_2'] = 'identity'
    kwargs['conv_layer_3'] = 'conv'
    kwargs['conv_layer_4'] = 'identity'
    kwargs['conv_layer_5'] = 'identity'
    kwargs['conv_layer_6'] = 'conv'
    kwargs['conv_layer_7'] = 'identity'
    kwargs['conv_layer_8'] = 'identity'

    Model(**kwargs)


if __name__ == '__main__':
    main()
