#!/usr/bin/env python

from exchange_data.models.resnet.model import ModelTrainer
from pathlib import Path
from tensorboard.plugins.hparams import api as hp

import alog
import click
import logging
import tensorflow as tf

HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.0, 0.05))
HP_LRATE = hp.HParam('Learning Rate', hp.RealInterval(.00001, .0001))

METRIC_ACCURACY = 'accuracy'
# HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd','RMSprop']))

def run(run_name, hparams):
    run_dir = f'{Path.home()}/.exchange-data/models/resnet_params/{run_name}'

    with tf.summary.create_file_writer(run_dir).as_default():
        params = {
            'batch_size': 1,
            'checkpoint_steps': 1000,
            'clear': True,
            'directory': 'tune',
            'epochs': 3,
            'eval_span': '2m',
            'eval_steps': '30s',
            'export_model': False,
            'frame_width': 224,
            'interval': '1m',
            'learning_rate_decay': 0,
            'max_steps': 21600,
            'seed': 216,
            'steps_epoch': '10m',
            'window_size': '3s',
        }


        hp.hparams(hparams)  # record the values used in this trial
        alog.info(hparams)

        hparams = {**params, **hparams}

        alog.info(alog.pformat(hparams))

        model = ModelTrainer(**hparams)
        accuracy = model.run().get('accuracy')
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)


@click.command()
# @click.option('--batch-size', '-b', type=int, default=1)
# @click.option('--checkpoint-steps', '-s', type=int, default=200)
# @click.option('--epochs', '-e', type=int, default=10)
# @click.option('--eval-span', type=str, default='20m')
# @click.option('--eval-steps', type=str, default='15s')
# @click.option('--frame-width', type=int, default=224)
# @click.option('--interval', '-i', type=str, default='1m')
# @click.option('--learning-rate', '-l', type=float, default=0.3e-4)
# @click.option('--learning-rate-decay', default=5e-3, type=float)
# @click.option('--max_steps', '-m', type=int, default=6 * 60 * 60)
# @click.option('--seed', type=int, default=6*6*6)
# @click.option('--steps-epoch', default='1m', type=str)
# @click.option('--window-size', '-w', default='3s', type=str)
# @click.option('--clear', '-c', is_flag=True)
def main(**kwargs):
    logging.getLogger('tensorflow').setLevel(logging.INFO)
    session_num = 0
    session_limit = 100

    while session_num <= session_limit:
        learning_rate = HP_LRATE.domain.sample_uniform()

        for l in range(0, 2):
            dropout_rate = HP_DROPOUT.domain.sample_uniform()
            hparams = dict(
                dropout_rate=dropout_rate,
                learning_rate=learning_rate,
            )

            run_name = "run-%d" % session_num
            print('--- Starting trial: %s' % run_name)

            alog.info(alog.pformat(hparams))
            try:
                run(run_name, hparams)
            except Exception as e:
                alog.info(e)
                pass

            session_num += 1


if __name__ == '__main__':
    main()
