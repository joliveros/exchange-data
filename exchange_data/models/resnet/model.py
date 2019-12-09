#!/usr/bin/env python
import tensorflow as tf

from exchange_data import settings
from exchange_data.emitters import Messenger
from exchange_data.tfrecord.dataset_query import dataset
from pathlib import Path
from pytimeparse.timeparse import timeparse
from tensorflow_core.python.keras.estimator import model_to_estimator
from tensorflow_estimator.python.estimator.run_config import RunConfig
from tensorflow_estimator.python.estimator.training import TrainSpec, EvalSpec, train_and_evaluate
import alog
import click
import logging
import shutil


Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
GlobalAveragePooling2D = tf.keras.layers.GlobalAveragePooling2D
Input = tf.keras.Input
LSTM = tf.keras.layers.LSTM
Reshape = tf.keras.layers.Reshape
ResNet50 = tf.keras.applications.ResNet50
Sequential = tf.keras.models.Sequential
SGD = tf.keras.optimizers.SGD


def Model(learning_rate, frame_width):
    model = Sequential()
    model.add(Input(shape=(frame_width, frame_width, 3)))

    base = ResNet50(include_top=False, weights=None, classes=3)
    for layer in base.layers:
        layer.trainable = True

    model.add(base)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.2))
    model.add(Dense(72, activation='relu'))
    model.add(Reshape((72, 1)))
    model.add(LSTM(3, return_sequences=False))
    # model.add(LSTM(24, return_sequences=False))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=SGD(lr=learning_rate, decay=5e-3),
                  metrics=['accuracy'])
    return model


class ModelTrainer(Messenger):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs

    def done(self):
        self.publish('resnet_trainer_done', '')

    def run(self):
        self._run(**self.kwargs)

    def _run(
            self,
            batch_size,
            checkpoint_steps,
            epochs,
            eval_span,
            eval_steps,
            frame_width,
            interval,
            learning_rate,
            max_steps,
            min_std_dev,
            steps_epoch,
            stddev_group_interval,
            window_size,
        ):
        model = Model(learning_rate, frame_width)

        model.summary()

        model_dir = f'{Path.home()}/.exchange-data/models/resnet'

        run_config = RunConfig(
            save_checkpoints_secs=timeparse(steps_epoch) * epochs
        )

        resnet_estimator = model_to_estimator(
            keras_model=model, model_dir=model_dir,
            checkpoint_format='saver',
            config=run_config,
        )

        train_spec = TrainSpec(
            input_fn=lambda: dataset(
                batch_size = batch_size,
                epochs=epochs,
                frame_width=frame_width,
                interval=interval,
                min_std_dev=min_std_dev,
                steps_epoch=steps_epoch,
                use_volatile_ranges=True,
                window_size=window_size,
                stddev_group_interval=stddev_group_interval,
            )
        )

        eval_spec = EvalSpec(
            input_fn=lambda: dataset(
                batch_size=batch_size,
                epochs=1,
                frame_width=frame_width,
                interval=eval_span,
                min_std_dev=min_std_dev,
                steps_epoch=steps_epoch,
                use_volatile_ranges=True,
                window_size=window_size,
                stddev_group_interval=stddev_group_interval,
            ),
            steps=timeparse(eval_steps),
            throttle_secs=timeparse(steps_epoch) * epochs
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
@click.option('--min-std-dev', type=float, default=0.0)
@click.option('--max_steps', '-m', type=int, default=6 * 60 * 60)
@click.option('--steps-epoch', default='1m', type=str)
@click.option('--window-size', '-w', default='3s', type=str)
@click.option('--stddev-group-interval', default='15s', type=str)
def main(**kwargs):
    logging.getLogger('tensorflow').setLevel(logging.INFO)
    trainer = ModelTrainer(**kwargs)
    trainer.run()


if __name__ == '__main__':
    main()
