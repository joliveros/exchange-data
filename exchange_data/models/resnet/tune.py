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
from exchange_data.utils import DateTimeUtils

HP_LRATE = hp.HParam('Learning Rate', hp.RealInterval(0.00001, 0.0001))
HP_EPSILON = hp.HParam('Epsilon', hp.RealInterval(0.09, 0.1064380))

METRIC_ACCURACY = 'accuracy'
# HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd','RMSprop']))
SYMBOL = 'KAVABNB'

backtest = BackTest(**{
    'start_date': DateTimeUtils.parse_datetime_str('2020-06-30 '
                                                   '23:31:00'),
    'end_date': DateTimeUtils.parse_datetime_str('2020-07-01 00:53:00'),
    'database_name': 'binance',
    'depth': 40,
    'interval': '3h',
    'plot': True,
    'sequence_length': 48,
    'symbol': SYMBOL,
    'volume_max': 10000.0,
})


def run(trial: Trial):
    backtest.trial = trial

    tf.keras.backend.clear_session()

    run_dir = f'{Path.home()}/.exchange-data/models/{SYMBOL}_params/' \
              f'{trial.number}'

    hparams = dict(
        # take_ratio=trial.suggest_float('take_ratio', 0.5005, 0.502),
        expected_position_length=trial.suggest_int(
            'expected_position_length', 1, 48),
        # epochs=trial.suggest_int('take_ratio', 1, 10),
    )

    labeled_count = convert(**{
        'dataset_name': f'{SYMBOL}_default',
        'expected_position_length': hparams.get('expected_position_length'),
        'labeled_ratio': 0.5,
        'min_change': 0.05,
        'sequence_length': 48
    })

    with tf.summary.create_file_writer(run_dir).as_default():
        params = {
         'take': labeled_count,
         'take_ratio': 0.5,
         'batch_size': 20,
         'clear': True,
         'directory': trial.number,
         'symbol': SYMBOL,
         'epochs': 10,
         'export_model': True,
         'learning_rate': 1.0e-5,
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

        time.sleep(5)

        backtest.test()

        if backtest.capital == 1.0:
            return 0.0

        return backtest.capital


@click.command()
def main(**kwargs):

    logging.getLogger('tensorflow').setLevel(logging.INFO)
    session_limit = 1000

    study = optuna.create_study(direction='maximize')

    study.optimize(run, n_trials=session_limit)





if __name__ == '__main__':
    main()
