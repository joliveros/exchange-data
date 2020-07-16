#!/usr/bin/env python

from pathlib import Path
from pytimeparse.timeparse import timeparse
from tensorflow_estimator.python.estimator.keras import model_to_estimator
from tensorflow_estimator.python.estimator.run_config import RunConfig
from tensorflow_estimator.python.estimator.training import TrainSpec, EvalSpec, train_and_evaluate

import alog
import click
import logging
import os
import shutil
import tensorflow as tf

from exchange_data.models.resnet.model import Model


class ModelTrainer(object):
    def __init__(self, **kwargs):
        alog.info(alog.pformat(kwargs))
        self.kwargs = kwargs

    def done(self):
        self.publish('resnet_trainer_done', '')

    def run(self):
        return self._run(**self.kwargs)

    def _run(
            self,
            train_df,
            eval_df,
            batch_size,
            clear,
            directory,
            epochs,
            export_model,
            learning_rate,
            levels,
            sequence_length,
            seed=None,
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
            **kwargs
        )

        model.summary()

        run_config = RunConfig(
            save_checkpoints_secs=timeparse('3m'),
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

        def iterate_rows(_df):
            def _rows():
                for i in range(0, len(_df)):
                    row = _df.iloc[i]
                    yield row['orderbook_img'], row['expected_position']

            return _rows

        def train_ds():
            return tf.data.Dataset.from_generator(
                iterate_rows(train_df),
                (tf.float32, tf.int32),
                (tf.TensorShape([sequence_length, levels * 2, 1]),
                 tf.TensorShape([]))
            ) \
                .cache() \
                .batch(batch_size).repeat(epochs)

        def eval_ds():
            return tf.data.Dataset.from_generator(
                iterate_rows(eval_df),
                (tf.float32, tf.int32),
                (tf.TensorShape([sequence_length, levels * 2, 1]),
                 tf.TensorShape([]))
            ) \
                .batch(1)


        # for frame, expected_position in tfds.as_numpy(train_ds()):
        #     alog.info(frame)
        #     alog.info(frame.shape)
        #     alog.info(expected_position)
        #     raise Exception()

        train_spec = TrainSpec(
            input_fn=train_ds,
        )

        eval_spec = EvalSpec(
            input_fn=eval_ds,
            start_delay_secs=timeparse('3m'),
            steps=timeparse('16m'),
            throttle_secs=timeparse('3m')
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
                         f'{symbol}_export'

            try:
                os.mkdir(export_dir)
            except:
                pass

            exp_path = resnet_estimator.export_saved_model(export_dir,
                                                serving_input_receiver_fn)

            result['exported_model_path'] = exp_path.decode()

        return result


@click.command()
@click.option('--batch-size', '-b', type=int, default=1)
@click.option('--levels', type=int, default=40)
@click.option('--sequence-length', type=int, default=48)
@click.option('--take', type=int, default=1000)
@click.option('--take-ratio', type=float, default=0.5)
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
