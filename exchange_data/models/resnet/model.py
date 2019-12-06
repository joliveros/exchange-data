#!/usr/bin/env python

from exchange_data import settings
from exchange_data.emitters import Messenger
from exchange_data.tfrecord.dataset import dataset
from pathlib import Path
from pytimeparse.timeparse import timeparse
from tensorflow_core.python.keras.estimator import model_to_estimator
from tensorflow_estimator.python.estimator.run_config import RunConfig
from tensorflow_estimator.python.estimator.training import TrainSpec, EvalSpec, train_and_evaluate

import alog
import click
import shutil
import tensorflow as tf

Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
GlobalAveragePooling2D = tf.keras.layers.GlobalAveragePooling2D
Input = tf.keras.Input
LSTM = tf.keras.layers.LSTM
Reshape = tf.keras.layers.Reshape
ResNet50 = tf.keras.applications.ResNet50
Sequential = tf.keras.models.Sequential
SGD = tf.keras.optimizers.SGD



def Model(learning_rate, frame_size):
    model = Sequential()
    model.add(Input(shape=(frame_size, frame_size, 3)))

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
        while True:
            self._run(**self.kwargs)

    def _run(
            self,
            batch_size,
            checkpoint_steps,
            clear,
            epochs,
            eval_span,
            frame_size,
            learning_rate,
            max_steps,
            save_checkpoint_secs,
        ):
        model = Model(learning_rate, frame_size)

        model.summary()

        model_dir = f'{Path.home()}/.exchange-data/models/resnet'

        if clear:
            try:
                shutil.rmtree(model_dir)
            except Exception:
                pass

        run_config = RunConfig(
            # save_checkpoints_steps=checkpoint_steps,
            save_checkpoints_secs=save_checkpoint_secs
        )
        resnet_estimator = model_to_estimator(
            keras_model=model, model_dir=model_dir,
            # checkpoint_format='checkpoint',
            checkpoint_format='saver',
            config=run_config,
        )

        eval_span = timeparse(eval_span)

        train_spec = TrainSpec(
            input_fn=lambda: dataset(batch_size, epochs).skip(eval_span),
        )

        eval_spec = EvalSpec(
            input_fn=lambda: dataset(batch_size, 1).take(eval_span),
            steps=eval_span,
            throttle_secs=60
            # hooks=[ProfitAndLossHook(resnet_estimator)]
        )

        train_and_evaluate(resnet_estimator, train_spec, eval_spec)

        # self.done()

        alog.info('#### DONE ####')


@click.command()
@click.option('--epochs', '-e', type=int, default=10)
@click.option('--max_steps', '-m', type=int, default=6 * 60 * 60)
@click.option('--batch-size', '-b', type=int, default=1)
@click.option('--learning-rate', '-l', type=float, default=0.3e-4)
@click.option('--clear', '-c', is_flag=True)
@click.option('--eval-span', type=str, default='20m')
@click.option('--checkpoint-steps', '-s', type=int, default=200)
@click.option('--save-checkpoint-secs', '-secs', type=int, default=200)
@click.option('--frame-size', type=int, default=224)
def main(**kwargs):
    tf.compat.v1.logging.set_verbosity(settings.LOG_LEVEL)
    trainer = ModelTrainer(**kwargs)
    trainer.run()


if __name__ == '__main__':
    main()
