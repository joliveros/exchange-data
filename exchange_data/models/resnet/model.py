#!/usr/bin/env python

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
    num_lstm,
    lstm_size,
    depth,
    sequence_length,
    batch_size,
    num_dense=3,
    dense_width=240,
    include_last=False,
    input_shape=None,
    learning_rate=5e-5,
    num_categories=2,
    **kwargs
):
    alog.info(alog.pformat((dense_width, kwargs)))
    tf.compat.v1.experimental.output_all_intermediates(True)

    if not input_shape:
        input_shape = (batch_size, sequence_length, depth * 2, 1)

    inputs = Input(shape=input_shape)

    conv = TimeDistributed(ResNetTS(
        input_shape[1:],
        **kwargs
    ).model)(inputs)

    alog.info(conv.shape)

    for l in range(num_lstm):
        conv = LSTM(lstm_size, return_sequences=l < num_lstm - 1)(conv)
        alog.info(l)
        alog.info(conv.shape)

    dense = Flatten()(conv)

    if num_dense > 0:
        for i in range(num_dense):
            dense = Dense(dense_width)(dense)

    dense_out = Dense(
        num_categories, activation='softmax',
        # use_bias=True,
        # bias_initializer=tf.keras.initializers.Constant(value=[0.0, 1.0])
    )

    dense_out.trainable = True

    out = dense_out(dense)

    alog.info(out.shape)

    # if include_last:
    #     out = dense_out(dense)
    # else:
    #     out = dense

    alog.info(out.shape)

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


class ResNetTS:
    def __init__(
        self,
        input_shape,
        gap_enabled=True,
        base_filter_size=4,
        block_filter_factor=2,
        block_kernel=3,
        kernel_size=2,
        max_pooling_kernel=2,
        max_pooling_strides=2,
        num_categories=2,
        num_conv=3,
        padding=3,
        strides=1,
        **kwargs
    ):
        alog.info(input_shape)

        input = Input(input_shape)

        alog.info(input)

        self.bn_axis = 3

        conv = tf.keras.layers.ZeroPadding2D(padding=(padding, padding))(input)
        alog.info(conv.shape)
        conv = Conv2D(filters=base_filter_size,
                      kernel_size=(kernel_size, kernel_size),
                      strides=(strides, strides),
                      padding='valid', kernel_initializer='he_normal')(conv)

        conv = tf.keras.layers.BatchNormalization(axis=self.bn_axis)(conv)
        conv = Activation('relu')(conv)

        conv = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv)
        # conv = tf.keras.layers.MaxPooling2D(
        #     (max_pooling_kernel, max_pooling_kernel),
        #     strides=(max_pooling_strides, max_pooling_strides))(conv)

        alog.info(conv.shape)

        filters = [base_filter_size, base_filter_size, base_filter_size]
        kernel_size = (block_kernel, block_kernel)

        filters2 = [filter * 2 for filter in filters]
        filters3 = [filter * 4 for filter in filters]
        filters4 = [filter * 8 for filter in filters]

        convs = [
            lambda conv: self.conv_block(conv, kernel_size, filters, [1, 1]),
            lambda conv: self.identity_block(conv, kernel_size, filters),
            lambda conv: self.identity_block(conv, kernel_size, filters),
            lambda conv: self.conv_block(conv, kernel_size, filters2),
            lambda conv: self.identity_block(conv, kernel_size, filters2),
            lambda conv: self.identity_block(conv, kernel_size, filters2),
            lambda conv: self.identity_block(conv, kernel_size, filters2),
            lambda conv: self.conv_block(conv, kernel_size, filters3),
            lambda conv: self.identity_block(conv, kernel_size, filters3),
            lambda conv: self.identity_block(conv, kernel_size, filters3),
            lambda conv: self.identity_block(conv, kernel_size, filters3),
            lambda conv: self.identity_block(conv, kernel_size, filters3),
            lambda conv: self.identity_block(conv, kernel_size, filters3),
            lambda conv: self.conv_block(conv, kernel_size, filters4),
            lambda conv: self.identity_block(conv, kernel_size, filters4),
            lambda conv: self.identity_block(conv, kernel_size, filters4),
            lambda conv: self.conv_block(conv, kernel_size, filters4),
            lambda conv: self.identity_block(conv, kernel_size, filters4),
            lambda conv: self.identity_block(conv, kernel_size, filters4),
        ]

        if num_conv > 0:
            convs = convs[:num_conv]

        for _conv in convs:
            alog.info(conv)
            conv = _conv(conv)

        alog.info(conv.shape)

        if gap_enabled:
            output = GlobalAveragePooling2D()(conv)
        else:
            output = Flatten()(conv)

        # alog.info(output.shape)
        self.model = tf.keras.models.Model(inputs=input, outputs=output)

        for layer in self.model.layers:
            layer.trainable = True

        self.model.summary()

    def conv_block(self, input_tensor, kernel_size, filters, strides=[2, 2]):
        conv = Conv2D(filters=filters[0], kernel_size=[1, 1], strides=strides,
                      kernel_initializer='he_normal', padding='valid')(input_tensor)
        conv = tf.keras.layers.BatchNormalization(axis=self.bn_axis)(conv)
        conv = Activation('relu')(conv)

        conv = Conv2D(filters=filters[1], kernel_size=kernel_size,
               padding='same', kernel_initializer='he_normal')(conv)
        conv = tf.keras.layers.BatchNormalization(axis=self.bn_axis)(conv)
        conv = Activation('relu')(conv)

        conv = Conv2D(filters=filters[2], padding='valid', kernel_size=[1, 1],
                      kernel_initializer='he_normal')(conv)

        conv = tf.keras.layers.BatchNormalization(axis=self.bn_axis)(conv)

        alog.info(conv.shape)

        alog.info(kernel_size)

        short_cut = Conv2D(filters=filters[2], kernel_size=[1, 1], strides=strides,
                           padding='valid', kernel_initializer='he_normal')(input_tensor)

        short_cut = tf.keras.layers.BatchNormalization(axis=self.bn_axis)(short_cut)

        alog.info(short_cut.shape)

        output = tf.keras.layers.add([conv, short_cut])
        output = Activation('relu')(output)

        return output

    def identity_block(self, input_tensor, kernel_size, filters, strides=[1, 1]):
        filters1, filters2, filters3 = filters

        x = Conv2D(filters1, (1, 1),
                          kernel_initializer='he_normal', padding='valid')(input_tensor)
        x = BatchNormalization(axis=self.bn_axis)(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size, strides,
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
    Model(**kwargs)


if __name__ == '__main__':
    main()
