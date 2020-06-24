#!/usr/bin/env python
import time

import optuna
from optuna import Trial

from exchange_data.models.resnet.model import ModelTrainer
from pathlib import Path
from tensorboard.plugins.hparams import api as hp

import alog
import click
import logging
import tensorflow as tf

HP_LRATE = hp.HParam('Learning Rate', hp.RealInterval(0.00001, 0.0001))
HP_EPSILON = hp.HParam('Epsilon', hp.RealInterval(0.09, 0.1064380))

METRIC_ACCURACY = 'accuracy'
# HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd','RMSprop']))


def run(trial: Trial):
    tf.keras.backend.clear_session()

    run_dir = f'{Path.home()}/.exchange-data/models/resnet_params/' \
              f'{trial.number}'

    hparams = dict(
        learning_rate=trial.suggest_float('learning_rate', 0.000001, 0.0001),
    )

    with tf.summary.create_file_writer(run_dir).as_default():
        params = {
         'batch_size': 20,
         'clear': True,
         'directory': trial.number,
         'epochs': 500,
         'export_model': False,
         'learning_rate': hparams.get('learning_rate'),
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

        return accuracy


@click.command()
def main(**kwargs):
    logging.getLogger('tensorflow').setLevel(logging.INFO)
    session_limit = 1000

    study = optuna.create_study(direction='maximize')

    study.optimize(run, n_trials=session_limit)





if __name__ == '__main__':
    main()
