#!/usr/bin/env python

import time
from argparse import Namespace

from optuna import Trial

from baselines.run import train
from tensorboard.plugins.hparams import api as hp

import alog
import click
import logging
import tensorflow as tf
import optuna

from exchange_data.utils import DateTimeUtils

HP_LRATE = hp.HParam('Learning Rate', hp.RealInterval(0.00001, 0.0003))
HP_REWARD_RATIO = hp.HParam('Reward Ratio', hp.RealInterval(0.99445, 1.0))
HP_EPSILON = hp.HParam('Epsilon', hp.RealInterval(0.09, 0.1064380))
HP_GAIN_DELAY = hp.HParam('gain_delay', hp.IntInterval(30, 360))
HP_FLAT_REWARD = hp.HParam('Flat Reward', hp.RealInterval(0.55250, 0.55598))

METRIC_ACCURACY = 'accuracy'
# HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd','RMSprop']))


def run(trial: Trial):
    tf.keras.backend.clear_session()

    run_name = str(int(time.time() * 1000))

    steps = 2000000

    hparams = dict(
        # kernel_dim=trial.suggest_categorical('kernel_dim', [
        #     2, 4, 8, 16, 32, 64, 128, 256
        # ])
        # lr=trial.suggest_float('lr', 0.001, 0.01),
        # min_change=trial.suggest_float('min_change', 3.0, 14.0),
        # min_change=12.0,
        flat_reward=trial.suggest_float('flat_reward', 0.001, 0.06),
        # reward_ratio=trial.suggest_float('reward_ratio', 0.001, 1.0),
        # step_reward_ratio=trial.suggest_float('step_reward_ratio', 0.001, 1.0),
        # step_reward=trial.suggest_float('step_reward', 0.01, 1.0),
        # max_loss=trial.suggest_float('max_loss', -0.02, -0.0001),
        # gain_delay=trial.suggest_float('gain_delay', 200, steps/2)
        # max_negative_pnl_delay=trial.suggest_int('max_negative_pnl_delay',
        #                                          190, 240),
        # max_negative_pnl=trial.suggest_float('max_negative_pnl', -0.006,
        #                                      -0.002)
    )

    # gain_delay = hparams.get('gain_delay')
    # expected_gain = 101
    # gain_per_step = ((expected_gain/100) - 1) / (steps - gain_delay)

    args = Namespace(
        # gain_delay=gain_delay,
        # gain_per_step=gain_per_step,
        alg='a2c',
        directory_name='default',
        env='tf-orderbook-v0',
        env_type=None,
        flat_reward=hparams.get('flat_reward'),
        gamestate=None,
        leverage=1.0,
        log_path=None,
        max_frames=48,
        max_loss=-0.01,
        max_negative_pnl=-0.0002648181835761804 * 2,
        max_negative_pnl_delay=0,
        max_steps=steps,
        levels=40,
        min_change=2.0,
        network='nasnet',
        num_env=1,
        num_timesteps=steps,
        play=False,
        reward_ratio=1.0,
        reward_scale=1.0,
        run_name=run_name,
        save_model=True,
        save_path=None,
        save_video_interval=0,
        save_video_length=200,
        seed=31583,
        step_reward=1.0,
        step_reward_ratio=1.0,
        trial=trial,
    )
    extra_args = {
        'batch_size': 1,
        'epsilon': 1e-7,
        'hparams': hparams,
        'kernel_dim': 4,
        'log_interval': 100,
        'lr': 0.01,
        'nsteps': 12,
    }

    model, env = train(args, extra_args)

    # try:
    #     model, env = train(args, extra_args)
    # except:
    #     return 0.0

    return model.capital


@click.command()
def main(**kwargs):
    physical_devices = tf.config.list_physical_devices('GPU')

    tf.config.set_logical_device_configuration(
        physical_devices[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=100),
         tf.config.LogicalDeviceConfiguration(memory_limit=100)])

    logical_devices = tf.config.list_logical_devices('GPU')

    assert len(logical_devices) == len(physical_devices) + 1
    logging.getLogger('tensorflow').setLevel(logging.INFO)
    session_limit = 1000

    study = optuna.create_study(direction='maximize')

    study.optimize(run, n_trials=session_limit)


if __name__ == '__main__':
    main()
