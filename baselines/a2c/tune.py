#!/usr/bin/env python

from argparse import Namespace
from baselines.run import train
from optuna import Trial

import alog
import click
import logging
import optuna
import tensorflow as tf
import tgym.envs
import time


def run(trial: Trial, **kwargs):
    tf.keras.backend.clear_session()

    run_name = str(trial.number)

    steps = 5e5

    hparams = dict(
        # kernel_dim=trial.suggest_categorical('kernel_dim', [
        #     2, 4, 8, 16, 32, 64, 128, 256
        # ])
        # lr=trial.suggest_float('lr', 0.001, 0.01),
        # min_change=trial.suggest_float('min_change', 3.0, 14.0),
        # min_change=12.0,
        # flat_reward=trial.suggest_float('flat_reward', 0.001, 0.06),
        # reward_ratio=trial.suggest_float('reward_ratio', 0.001, 1.0),
        # step_reward_ratio=trial.suggest_float('step_reward_ratio', 0.001, 1.0),
        # step_reward=trial.suggest_float('step_reward', 0.01, 1.0),
        # max_loss=trial.suggest_float('max_loss', -0.02, -0.0001),
        # gain_delay=trial.suggest_float('gain_delay', 200, steps/2)
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
        env='orderbook-frame-env-v0',
        env_type=None,
        flat_reward=1.0,
        gamestate=None,
        leverage=1.0,
        log_path=None,
        max_frames=48,
        max_loss=-0.5,
        max_negative_pnl=-0.5,
        max_steps=steps,
        levels=40,
        min_change=0.0001,
        network='resnet',
        num_env=1,
        num_timesteps=steps,
        play=False,
        reward_ratio=1e-4,
        run_name=run_name,
        save_model=True,
        save_path=None,
        save_video_interval=0,
        save_video_length=200,
        seed=31583,
        step_reward=1.0,
        step_reward_ratio=1e-4,
        trial=trial,
        **kwargs
    )
    extra_args = {
        'hparams': hparams,
        'log_interval': 100,
        'lr': 0.0000001,
        'nsteps': 3
    }

    model, env = train(args, extra_args)

    return model.capital


@click.command()
@click.option('--database_name', '-d', default='binance_futures', type=str)
@click.option('--depth', default=72, type=int)
@click.option('--group-by', '-g', default='30s', type=str)
@click.option('--interval', '-i', default='10m', type=str)
@click.option('--max-volume-quantile', '-m', default=0.99, type=float)
@click.option('--offset-interval', '-o', default='0h', type=str)
@click.option('--round-decimals', '-D', default=4, type=int)
@click.option('--sequence-length', '-l', default=24, type=int)
@click.option('--summary-interval', '-s', default=1, type=int)
@click.option('--window-size', '-w', default='2m', type=str)
@click.argument('symbol', type=str)
def main(**kwargs):
    physical_devices = tf.config.list_physical_devices('GPU')

    if len(physical_devices):
        tf.config.set_logical_device_configuration(
            physical_devices[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=256),
             tf.config.LogicalDeviceConfiguration(memory_limit=256)])

        logical_devices = tf.config.list_logical_devices('GPU')

        assert len(logical_devices) == len(physical_devices) + 1

    logging.getLogger('tensorflow').setLevel(logging.INFO)
    session_limit = 1000

    study = optuna.create_study(direction='maximize')

    study.optimize(lambda trial: run(trial, **kwargs), n_trials=session_limit)


if __name__ == '__main__':
    main()
