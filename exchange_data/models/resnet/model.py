#!/usr/bin/env python

import alog
import click
import tensorflow as tf

Activation = tf.keras.layers.Activation
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
    depth,
    sequence_length,
    num_conv=2,
    filters=1,
    base_filter_size=16,
    inception_units=1,
    lstm_units=8,
    relu_alpha=0.01,
    learning_rate=5e-5,
    num_categories=2,
    **kwargs
):
    input_shape = (sequence_length, depth * 2, 1)
    alog.info(input_shape)

    inputs = Input(shape=input_shape)

    filters = filters * base_filter_size

    conv = TimeDistributed(ResNetTS(input_shape[1:], filters, num_categories,
                                    num_conv
                                    ))(inputs)

    inception_units = inception_units * base_filter_size
    lstm_units = lstm_units * base_filter_size

    alog.info(conv.shape)

    # build the inception module
    convsecond_1 = Conv1D(inception_units, 1, padding='same')(conv)
    convsecond_1 = LeakyReLU(alpha=relu_alpha)(convsecond_1)
    convsecond_1 = Conv1D(inception_units, 3, padding='same')(convsecond_1)
    convsecond_1 = LeakyReLU(alpha=relu_alpha)(convsecond_1)

    alog.info(convsecond_1.shape)

    convsecond_2 = Conv1D(inception_units, 1, padding='same')(conv)
    convsecond_2 = LeakyReLU(alpha=relu_alpha)(convsecond_2)
    convsecond_2 = Conv1D(inception_units, 5, padding='same')(convsecond_2)
    convsecond_2 = LeakyReLU(alpha=relu_alpha)(convsecond_2)

    alog.info(convsecond_2.shape)

    convsecond_3 = tf.keras.layers.MaxPool1D(3, strides=1, padding='same')(
        conv)
    convsecond_3 = Conv1D(inception_units, 1, padding='same')(convsecond_3)
    convsecond_3 = LeakyReLU(alpha=relu_alpha)(convsecond_3)

    alog.info(convsecond_3.shape)

    convsecond_output = tf.keras.layers.concatenate(
        [convsecond_1, convsecond_2, convsecond_3], axis=2)

    alog.info(convsecond_output.shape)

    # build the last LSTM layer
    # lstm_out = LSTM(lstm_units, return_sequences=True, stateful=False,
    #                 recurrent_activation='sigmoid')(
    #     convsecond_output)

    # lstm_out = LSTM(lstm_units, return_sequences=True, stateful=False,
    #                 recurrent_activation='sigmoid')(
    #     lstm_out)
    #
    # lstm_out = LSTM(lstm_units, return_sequences=True, stateful=False,
    #                 recurrent_activation='sigmoid')(convsecond_output)

    lstm_out = LSTM(lstm_units, return_sequences=False, stateful=False,
                    recurrent_activation='sigmoid')(convsecond_output)

    alog.info(lstm_out.shape)
    alog.info(num_categories)

    dense_out = Dense(
        num_categories, activation='softmax',
        use_bias=True,
        bias_initializer=tf.keras.initializers.Constant(value=[0.0, 1.0])
    )

    out = dense_out(lstm_out)

    alog.info(out.shape)

    model = tf.keras.Model(inputs=inputs, outputs=out)

    model.compile(
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        optimizer=tf.keras.optimizers.Adadelta(learning_rate=learning_rate,
                                               clipnorm=1.0)
    )
    return model


def ResNetTS(input_shape, filters=64, num_categories=2, num_conv=2):
    if num_conv < 1:
        raise Exception()

    input = Input(input_shape)
    conv = input

    conv = Conv1D(filters=filters, kernel_size=1, padding='same')(input)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    for i in range(0, num_conv):
        conv = conv_block(filters, conv, i)

    gap = GlobalAveragePooling1D()(conv)

    alog.info(gap.shape)

    output = gap

    # output = Dense(num_categories, activation='softmax')(gap)

    alog.info(output.shape)

    return tf.keras.models.Model(inputs=input, outputs=output)


def conv_block(filters, last_conv, block_n):

    if block_n > 0:
        filters = filters * 2

    alog.info(last_conv.shape)
    alog.info((block_n, filters))

    conv = Conv1D(filters=filters, kernel_size=3, padding='same')(last_conv)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    conv = Conv1D(filters=filters, kernel_size=3, padding='same')(conv)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    conv = Conv1D(filters=filters, kernel_size=3, padding='same')(conv)
    conv = tf.keras.layers.BatchNormalization()(conv)

    alog.info(conv.shape)

    if last_conv.shape[-1] != filters:
        short_cut = Conv1D(filters=filters, kernel_size=1, padding='same')(
            last_conv)
        short_cut = tf.keras.layers.BatchNormalization()(short_cut)
    else:
        short_cut = tf.keras.layers.BatchNormalization()(last_conv)

    alog.info(short_cut.shape)

    output = tf.keras.layers.add([short_cut, conv])
    output = Activation('relu')(output)

    return output


@click.command()
@click.option('--batch-size', '-b', type=int, default=1)
@click.option('--depth', type=int, default=40)
@click.option('--sequence-length', type=int, default=48)
def main(**kwargs):
    Model(**kwargs)


if __name__ == '__main__':
    main()
