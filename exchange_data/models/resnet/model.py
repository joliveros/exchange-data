#!/usr/bin/env python

import alog
from tensorflow_estimator.python.estimator.run_config import RunConfig
from exchange_data import settings
from exchange_data.models.pnl_hook import ProfitAndLossHook
from exchange_data.tfrecord.dataset import dataset
from pathlib import Path
from pytimeparse.timeparse import timeparse
from tensorflow.python.keras import Input
from tensorflow.python.keras.api.keras import models
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.estimator import model_to_estimator
from tensorflow.python.keras.layers import Dense, Reshape, LSTM, GlobalAveragePooling2D, Dropout
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow_estimator.python.estimator.training import TrainSpec, EvalSpec, train_and_evaluate
import shutil
import click
import tensorflow as tf


def Model(learning_rate, frame_size):
    model = models.Sequential()
    model.add(Input(shape=(frame_size, frame_size, 3)))

    base = ResNet50(include_top=False, weights=None, classes=3)
    for layer in base.layers:
        layer.trainable = True

    model.add(base)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.2))
    model.add(Dense(96, activation='relu'))
    model.add(Reshape((96, 1)))
    model.add(LSTM(16, return_sequences=False))
    # model.add(LSTM(24, return_sequences=False))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=SGD(lr=learning_rate, decay=5e-3),
                  metrics=['accuracy'])
    return model


@click.command()
@click.option('--epochs', '-e', type=int, default=10)
@click.option('--batch-size', '-b', type=int, default=20)
@click.option('--learning-rate', '-l', type=float, default=0.3e-4)
@click.option('--clear', '-c', is_flag=True)
@click.option('--eval-span', type=str, default='20m')
@click.option('--checkpoint-steps', type=int, default=200)
@click.option('--frame-size', type=int, default=224)
def main(epochs, batch_size, clear, learning_rate, eval_span, checkpoint_steps,
        frame_size,
        **kwargs
    ):
    tf.compat.v1.logging.set_verbosity(settings.LOG_LEVEL)

    model = Model(learning_rate, frame_size)

    model.summary()

    model_dir = f'{Path.home()}/.exchange-data/models/resnet'

    if clear:
        try:
            shutil.rmtree(model_dir)
        except Exception:
            pass

    run_config = RunConfig(
        # save_checkpoints_steps=checkpoint_steps
        save_checkpoints_secs=60
    )
    resnet_estimator = model_to_estimator(
        keras_model=model, model_dir=model_dir,
        # checkpoint_format='checkpoint',
        checkpoint_format='saver',
        config=run_config,
    )

    eval_span = timeparse(eval_span)
    train_spec = TrainSpec(input_fn=lambda: dataset(batch_size, epochs).skip(eval_span),
                           max_steps=epochs * 6 * 60 * 60)
    eval_spec = EvalSpec(
        input_fn=lambda: dataset(batch_size, 1).take(eval_span),
        steps=eval_span,
        # hooks=[ProfitAndLossHook(resnet_estimator)]
    )

    train_and_evaluate(resnet_estimator, train_spec, eval_spec)


if __name__ == '__main__':
    main()
