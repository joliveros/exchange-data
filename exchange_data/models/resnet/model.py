#!/usr/bin/env python

from pandas import DataFrame
from pathlib import Path

from pytimeparse.timeparse import timeparse
from tensorflow_estimator.python.estimator.inputs.pandas_io import \
    pandas_input_fn
from tensorflow_estimator.python.estimator.keras import model_to_estimator
from tensorflow_estimator.python.estimator.run_config import RunConfig
from tensorflow_estimator.python.estimator.training import TrainSpec, EvalSpec, train_and_evaluate

import alog
import click
import logging
import shutil
import tensorflow as tf
import pandas as pd
import numpy as np

from exchange_data.tfrecord.processed_dataset import dataset

Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Flatten = tf.keras.layers.Flatten
GlobalAveragePooling2D = tf.keras.layers.GlobalAveragePooling2D
GlobalAveragePooling1D = tf.keras.layers.GlobalAveragePooling1D
Input = tf.keras.Input
LSTM = tf.keras.layers.LSTM
ConvLSTM2D = tf.keras.layers.ConvLSTM2D
Reshape = tf.keras.layers.Reshape
Sequential = tf.keras.models.Sequential
TimeDistributed = tf.keras.layers.TimeDistributed
Conv2D = tf.keras.layers.Conv2D
LeakyReLU = tf.keras.layers.LeakyReLU
MaxPooling2D = tf.keras.layers.MaxPooling2D

def Model(
    levels,
    sequence_length,
    batch_size=12,
    learning_rate=5e-5,
    num_categories=2,
    include_last=True,
    **kwargs
):
    input_shape = (sequence_length, levels * 2, 1)

    alog.info(input_shape)

    filters = 32

    inputs = Input(shape=input_shape)

    alog.info(inputs.shape)

    # build the convolutional block
    conv_first1 = Conv2D(filters, (1, 2), strides=(1, 2))(inputs)
    conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(filters, (4, 1), padding='same')(conv_first1)
    conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(filters, (4, 1), padding='same')(conv_first1)
    conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)

    alog.info(conv_first1.shape)

    conv_first1 = Conv2D(filters, (1, 2), strides=(1, 2))(conv_first1)
    conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(filters, (4, 1), padding='same')(conv_first1)
    conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(filters, (4, 1), padding='same')(conv_first1)
    conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)

    alog.info(conv_first1.shape)

    conv_first1 = Conv2D(filters, (1, 7))(conv_first1)
    conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(filters, (4, 1), padding='same')(conv_first1)
    conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(filters, (4, 1), padding='same')(conv_first1)
    conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)

    alog.info(conv_first1.shape)

    conv_first1 = Conv2D(filters, (1, 8))(conv_first1)
    conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(filters, (4, 1), padding='same')(conv_first1)
    conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(filters, (4, 1), padding='same')(conv_first1)
    conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)

    alog.info(conv_first1.shape)

    conv_first1 = Conv2D(filters, (1, 7))(conv_first1)
    conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(filters, (4, 1), padding='same')(conv_first1)
    conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(filters, (4, 1), padding='same')(conv_first1)
    conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)

    alog.info(conv_first1)
    # build the inception module
    convsecond_1 = Conv2D(64, (1, 1), padding='same')(conv_first1)
    convsecond_1 = LeakyReLU(alpha=0.01)(convsecond_1)
    convsecond_1 = Conv2D(64, (3, 1), padding='same')(convsecond_1)
    convsecond_1 = LeakyReLU(alpha=0.01)(convsecond_1)

    convsecond_2 = Conv2D(64, (1, 1), padding='same')(conv_first1)
    convsecond_2 = LeakyReLU(alpha=0.01)(convsecond_2)
    convsecond_2 = Conv2D(64, (5, 1), padding='same')(convsecond_2)
    convsecond_2 = LeakyReLU(alpha=0.01)(convsecond_2)

    convsecond_3 = MaxPooling2D((3, 1), strides=(1, 1), padding='same')(
        conv_first1)
    convsecond_3 = Conv2D(64, (1, 1), padding='same')(convsecond_3)
    convsecond_3 = LeakyReLU(alpha=0.01)(convsecond_3)

    convsecond_output = tf.keras.layers.concatenate(
        [convsecond_1, convsecond_2, convsecond_3], axis=3)

    alog.info(convsecond_output.shape)

    # use the MC dropout here
    conv_reshape = Reshape(
        (int(convsecond_output.shape[1]), int(convsecond_output.shape[3])))(
        convsecond_output)

    # build the last LSTM layer
    lstm_out = LSTM(64, return_sequences=False, stateful=False)(conv_reshape)

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


class ModelTrainer(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def done(self):
        self.publish('resnet_trainer_done', '')

    def run(self):
        return self._run(**self.kwargs)

    def _run(
            self,
            batch_size,
            clear,
            directory,
            epochs,
            export_model,
            learning_rate,
            levels,
            sequence_length,
            seed,
            symbol=None,
            **kwargs
        ):

        model_dir = f'{Path.home()}/.exchange-data/models/resnet/{directory}'

        if clear:
            try:
                shutil.rmtree(model_dir)
            except Exception:
                pass

        model = Model(
            levels=levels,
            sequence_length=sequence_length,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

        model.summary()

        run_config = RunConfig(
            save_checkpoints_secs=60*3,
            tf_random_seed=seed
        )

        resnet_estimator = model_to_estimator(
            keras_model=model,
            model_dir=model_dir,
            checkpoint_format='saver',
            config=run_config,
        )

        if symbol is None:
            symbol = directory

        train_spec = TrainSpec(
            input_fn=lambda: dataset(batch_size=batch_size, epochs=epochs,
                                     dataset_name=f'{symbol}_default'),
        )

        eval_spec = EvalSpec(
            input_fn=lambda: dataset(dataset_name=f'{symbol}_eval',
                                     batch_size=1,
                                     epochs=1),
            start_delay_secs=60,
            steps=timeparse('8m'),
            throttle_secs=60
        )

        result = train_and_evaluate(resnet_estimator, train_spec, eval_spec)[0]

        alog.info(result)

        def serving_input_receiver_fn():
            inputs = {
              'input_1': tf.compat.v1.placeholder(
                  tf.float32, [1, sequence_length, levels * 2, 1]
              ),
            }
            return tf.estimator.export.ServingInputReceiver(inputs, inputs)

        if export_model:
            export_dir = f'{Path.home()}/.exchange-data/models/' \
                         f'{directory}_export'

            resnet_estimator.export_saved_model(export_dir,
                                                serving_input_receiver_fn)

        return result


@click.command()
@click.option('--batch-size', '-b', type=int, default=1)
@click.option('--levels', type=int, default=40)
@click.option('--sequence-length', type=int, default=48)
@click.option('--epochs', type=int, default=500)
@click.option('--directory', type=str, default='default')
@click.option('--learning-rate', '-l', type=float, default=1e-5)
@click.option('--seed', type=int, default=6*6*6)
@click.option('--clear', '-c', is_flag=True)
@click.option('--export-model', is_flag=True)
def main(**kwargs):
    logging.getLogger('tensorflow').setLevel(logging.INFO)
    trainer = ModelTrainer(**kwargs)
    trainer.run()


if __name__ == '__main__':
    main()
