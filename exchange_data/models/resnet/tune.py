#!/usr/bin/env python
import time

import optuna
from optuna import Trial

from exchange_data.emitters.backtest import BackTest
from exchange_data.models.resnet.model import ModelTrainer
from pathlib import Path
from tensorboard.plugins.hparams import api as hp

import alog
import click
import logging
import tensorflow as tf

from exchange_data.tfrecord.concat_files import convert

HP_LRATE = hp.HParam('Learning Rate', hp.RealInterval(0.00001, 0.0001))
HP_EPSILON = hp.HParam('Epsilon', hp.RealInterval(0.09, 0.1064380))

METRIC_ACCURACY = 'accuracy'
# HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd','RMSprop']))
SYMBOL = 'SOLBNB'


def run(trial: Trial):
    tf.keras.backend.clear_session()

    run_dir = f'{Path.home()}/.exchange-data/models/{SYMBOL}_params/' \
              f'{trial.number}'

    hparams = dict(
        labeled_ratio=trial.suggest_float('labeled_ratio', 0.49, 0.55),
    )

    with tf.summary.create_file_writer(run_dir).as_default():
        convert(**{
            'dataset_name': f'{SYMBOL}_default',
            'expected_position_length': 1,
            'labeled_ratio': hparams.get('labeled_ratio'),
            'min_change': 0.01,
            'sequence_length': 48
        })

        params = {
         'batch_size': 20,
         'clear': True,
         'directory': trial.number,
         'symbol': SYMBOL,
         'epochs': 50,
         'export_model': True,
         'learning_rate': 1.0e-6,
         'levels': 40,
         'seed': 216,
         'sequence_length': 48
        }

        hp.hparams(hparams, trial_id=str(trial.number))  # record the values used in
        # this trial

        alog.info(hparams)

        hparams = {**params, **hparams}

        alog.info(alog.pformat(hparams))

        model = ModelTrainer(**hparams)
        result = model.run()
        accuracy = result.get('accuracy')
        global_step = result.get('global_step')

        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=global_step)


        result = BackTest(**{
            'database_name': 'binance',
            'depth': 40,
            'interval': '3h',
            'sequence_length': 48,
            'symbol': SYMBOL,
            'volume_max': 10000.0,
            'trial': trial
        })

        return result.capital


@click.command()
def main(**kwargs):
    logging.getLogger('tensorflow').setLevel(logging.INFO)
    session_limit = 1000

    study = optuna.create_study(direction='maximize')

    study.optimize(run, n_trials=session_limit)





if __name__ == '__main__':
    main()
