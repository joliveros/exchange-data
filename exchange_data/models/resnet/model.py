#!/usr/bin/env python
import shutil

import tensorflow as tf

from exchange_data import settings
from exchange_data.tfrecord.dataset import dataset
from exchange_data.utils import EventEmitterBase
from exchange_data.emitters import Messenger
from pathlib import Path
from pytimeparse.timeparse import timeparse
from tensorflow_core.python.keras.estimator import model_to_estimator
from tensorflow_estimator.python.estimator.run_config import RunConfig
from tensorflow_estimator.python.estimator.training import TrainSpec, EvalSpec, train_and_evaluate
import alog
import click
import logging


Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Flatten = tf.keras.layers.Flatten
GlobalAveragePooling2D = tf.keras.layers.GlobalAveragePooling2D
Input = tf.keras.Input
LSTM = tf.keras.layers.LSTM
Reshape = tf.keras.layers.Reshape
ResNet = tf.keras.applications.ResNet152V2
Sequential = tf.keras.models.Sequential
SGD = tf.keras.optimizers.SGD
Adam = tf.keras.optimizers.Adam
TimeDistributed = tf.keras.layers.TimeDistributed


def Model(
    batch_size,
    learning_rate=5e-5,
    frame_width=224,
    num_categories=3,
    learning_rate_decay=5e-3
):
    model = Sequential()
    model.add(Input(shape=(frame_width, frame_width, 3)))

    base = ResNet(
        include_top=False,
        classes=num_categories,
        pooling=None
    )

    model.add(base)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.1))
    model.add(Dense(num_categories, activation='softmax'))

    model.compile(
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        optimizer='adam'
    )

    return model


class ModelTrainer(Messenger):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def done(self):
        self.publish('resnet_trainer_done', '')

    def run(self):
        self._run(**self.kwargs)

    def _run(
            self,
            batch_size,
            clear,
            checkpoint_steps,
            epochs,
            eval_span,
            eval_steps,
            frame_width,
            interval,
            learning_rate,
            max_steps,
            learning_rate_decay,
            steps_epoch,
            window_size,
            seed
        ):
        model_dir = f'{Path.home()}/.exchange-data/models/resnet'

        if clear:
            try:
                shutil.rmtree(model_dir)
            except Exception:
                pass

        model = Model(
            batch_size=batch_size,
            learning_rate=learning_rate,
            frame_width=frame_width,
            learning_rate_decay=learning_rate_decay
        )

        model.summary()

        run_config = RunConfig(
            save_checkpoints_secs=540,
            tf_random_seed=seed
        )

        resnet_estimator = model_to_estimator(
            keras_model=model,
            model_dir=model_dir,
            checkpoint_format='saver',
            config=run_config,
        )

        train_spec = TrainSpec(
            input_fn=lambda: dataset(
                batch_size=batch_size,
                epochs=epochs,
            )
        )

        def eval_dataset(**kwargs):
            return dataset(batch_size=batch_size)

        eval_spec = EvalSpec(
            input_fn=lambda: eval_dataset(
                batch_size=batch_size,
                epochs=1,
                frame_width=frame_width,
                interval=eval_span,
                steps_epoch=eval_steps,
                window_size=window_size,
                use_volatile_ranges=True
            ),
            start_delay_secs=60*30,
            steps=timeparse(eval_steps)*2,
            throttle_secs=60*30
        )

        train_and_evaluate(resnet_estimator, train_spec, eval_spec)

        def serving_input_receiver_fn():
            inputs = {
              'input_1': tf.compat.v1.placeholder(
                  tf.float32, [None, frame_width, frame_width, 3]
              ),
            }
            return tf.estimator.export.ServingInputReceiver(inputs, inputs)

        resnet_estimator.export_saved_model(model_dir + '/saved', serving_input_receiver_fn)

        self.done()

        alog.info('#### DONE ####')


@click.command()
@click.option('--batch-size', '-b', type=int, default=1)
@click.option('--checkpoint-steps', '-s', type=int, default=200)
@click.option('--epochs', '-e', type=int, default=10)
@click.option('--eval-span', type=str, default='20m')
@click.option('--eval-steps', type=str, default='15s')
@click.option('--frame-width', type=int, default=224)
@click.option('--interval', '-i', type=str, default='1m')
@click.option('--learning-rate', '-l', type=float, default=0.3e-4)
@click.option('--learning-rate-decay', default=5e-3, type=float)
@click.option('--max_steps', '-m', type=int, default=6 * 60 * 60)
@click.option('--seed', type=int, default=6*6*6)
@click.option('--steps-epoch', default='1m', type=str)
@click.option('--window-size', '-w', default='3s', type=str)
@click.option('--clear', '-c', is_flag=True)
def main(**kwargs):
    logging.getLogger('tensorflow').setLevel(logging.INFO)
    trainer = ModelTrainer(**kwargs)
    trainer.run()


if __name__ == '__main__':
    main()
