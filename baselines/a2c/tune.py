#!/usr/bin/env python

from argparse import Namespace

from optuna import Trial

from baselines.run import train
from tensorboard.plugins.hparams import api as hp

import alog
import click
import logging
import tensorflow as tf
import optuna

HP_LRATE = hp.HParam('Learning Rate', hp.RealInterval(0.00001, 0.0003))
HP_REWARD_RATIO = hp.HParam('Reward Ratio', hp.RealInterval(0.99445, 1.0))
HP_EPSILON = hp.HParam('Epsilon', hp.RealInterval(0.09, 0.1064380))
HP_GAIN_DELAY = hp.HParam('gain_delay', hp.IntInterval(30, 360))
HP_FLAT_REWARD = hp.HParam('Flat Reward', hp.RealInterval(0.55250, 0.55598))

METRIC_ACCURACY = 'accuracy'
# HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd','RMSprop']))


def run(trial: Trial):
    run_name = f'run_{trial.number}'

    hparams = dict(
        # kernel_dim=trial.suggest_int('kernel_dim', 2, 128),
        # filters=trial.suggest_int('filters', 8, 128)
        reward_ratio=trial.suggest_float('reward_ratio', 0.0000001, 1.0),
        step_reward=trial.suggest_float('step_reward', 0.0001, 1.0)
    )

    # gain_delay = hparams.get('gain_delay')
    steps = 700
    # expected_gain = 150
    # gain_per_step = ((expected_gain/100) - 1) / (steps - gain_delay)

    args = Namespace(
        # gain_delay=gain_delay,
        # gain_per_step=gain_per_step,
        alg='a2c',
        directory_name='default',
        env='tf-orderbook-v0',
        env_type=None,
        flat_reward=0.0000062982,
        step_reward=hparams.get('step_reward'),
        gamestate=None,
        leverage=10.0,
        log_path=None,
        max_frames=5,
        max_steps=steps,
        min_change=7.4588,
        network='nasnet',
        num_env=1,
        num_timesteps=steps,
        play=False,
        reward_ratio=hparams.get('reward_ratio'),
        reward_scale=1.0,
        run_name=run_name,
        save_model=True,
        save_path=None,
        save_video_interval=0,
        save_video_length=200,
        seed=None
    )
    extra_args = {
        # 'epsilon': 0.090979,
        'lr': 0.00015753,
        'batch_size': 1,
        'epsilon': 1e-7,
        'filters': 13,
        'kernel_dim': 12,
        'log_interval': 100,
        'nsteps': 20,
        'hparams': hparams
    }

    try:
        model, env = train(args, extra_args)
    except:
        return 0.0

    return model.capital


@click.command()
def main(**kwargs):
    logging.getLogger('tensorflow').setLevel(logging.INFO)
    session_limit = 1000

    study = optuna.create_study(direction='maximize')

    study.optimize(run, n_trials=session_limit)


if __name__ == '__main__':
    main()
