#!/usr/bin/env python

import alog
import click
import tensorflow as tf

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
    levels,
    inception_units,
    filters,
    lstm_units,
    sequence_length,
    learning_rate=5e-5,
    num_categories=2,
    **kwargs
):
    input_shape = (sequence_length, levels * 2, 1)
    alog.info(input_shape)

    filters = filters * 16

    inputs = Input(shape=input_shape)

    alog.info(inputs.shape)

    filter_height = [2, ]
    last_layer_filter_height = 0
    last_conv = inputs

    while last_layer_filter_height != 1:
        for i in range(0, len(filter_height)):
            if last_layer_filter_height != 1:
                f = filter_height[i]
                if (last_layer_filter_height - f) < 0:
                    f = f - 1

                last_conv = conv_block(filters, last_conv, f)
                last_layer_filter_height = last_conv.shape[2]
                alog.info(last_conv.shape)

                if f not in filter_height:
                    break

    conv = last_conv

    inception_units = inception_units * 16
    lstm_units = lstm_units * 16

    alog.info(conv.shape)
    # build the inception module
    convsecond_1 = Conv2D(inception_units, (1, 1), padding='same')(conv)
    convsecond_1 = LeakyReLU(alpha=0.01)(convsecond_1)
    convsecond_1 = Conv2D(inception_units, (3, 1), padding='same')(convsecond_1)
    convsecond_1 = LeakyReLU(alpha=0.01)(convsecond_1)

    alog.info(convsecond_1.shape)

    convsecond_2 = Conv2D(inception_units, (1, 1), padding='same')(conv)
    convsecond_2 = LeakyReLU(alpha=0.01)(convsecond_2)
    convsecond_2 = Conv2D(inception_units, (5, 1), padding='same')(convsecond_2)
    convsecond_2 = LeakyReLU(alpha=0.01)(convsecond_2)

    alog.info(convsecond_2.shape)

    convsecond_3 = MaxPooling2D((3, 1), strides=(1, 1), padding='same')(
        conv)
    convsecond_3 = Conv2D(inception_units, (1, 1), padding='same')(convsecond_3)
    convsecond_3 = LeakyReLU(alpha=0.01)(convsecond_3)

    alog.info(convsecond_3.shape)

    convsecond_output = tf.keras.layers.concatenate(
        [convsecond_1, convsecond_2, convsecond_3], axis=3)

    alog.info(convsecond_output.shape)

    # use the MC dropout here
    conv_reshape = Reshape(
        (int(convsecond_output.shape[1]), int(convsecond_output.shape[3])))(
        convsecond_output)

    # build the last LSTM layer
    lstm_out = LSTM(lstm_units, return_sequences=False, stateful=False)(conv_reshape)

    alog.info(lstm_out.shape)
    alog.info(num_categories)
    out = Dense(num_categories, activation='softmax')(lstm_out)

    alog.info(out.shape)

    model = tf.keras.Model(inputs=inputs, outputs=out)

    model.compile(
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        optimizer=tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
    )
    return model


def conv_block(filters, last_conv, filter_height=2):
    conv = Conv2D(filters, (1, filter_height), strides=(1, filter_height))(
        last_conv)
    conv = LeakyReLU(alpha=0.01)(conv)
    conv = Conv2D(filters, (4, 1), padding='same')(conv)
    conv = LeakyReLU(alpha=0.01)(conv)
    conv = Conv2D(filters, (4, 1), padding='same')(conv)
    last_conv = LeakyReLU(alpha=0.01)(conv)
    return last_conv


@click.command()
@click.option('--batch-size', '-b', type=int, default=1)
@click.option('--levels', type=int, default=40)
@click.option('--sequence-length', type=int, default=48)
def main(**kwargs):
    Model(**kwargs)

if __name__ == '__main__':
    main()
