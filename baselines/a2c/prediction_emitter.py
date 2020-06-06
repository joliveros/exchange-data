#!/usr/bin/env python

import gzip
import logging
from argparse import Namespace
from collections import deque

from baselines.a2c.model import Model
from baselines.run import build_env
from exchange_data import settings
from exchange_data.channels import BitmexChannels
from exchange_data.emitters import Messenger
from exchange_data.trading import Positions
from tgym.envs.orderbook.ascii_image import AsciiImage
from pathlib import Path
from exchange_data import Measurement, NumpyEncoder, settings

import alog
import click
import json
import numpy as np
import tensorflow as tf


class TradeJob(object):
    def __init__(self, symbol: BitmexChannels, **kwargs):
        self.symbol = symbol
        self.job_name = f'trade_{self.symbol.value}'


class PredictionEmitter(Messenger, TradeJob):
    def __init__(self, symbol, depth, run_name, **kwargs):
        self.symbol = symbol

        super().__init__(symbol=symbol, decode=False, **kwargs)
        self.orderbook_channel = f'orderbook_img_frame_{symbol.value}_{depth}'
        self.saved_model = f'{Path.home()}/.exchange-data/models/a2c/model_export/{run_name}.h5'

        self.on(self.orderbook_channel, self.emit_prediction)

        env_args = Namespace(
            alg='a2c',
            batch_size=1,
            directory_name='default',
            env='tf-orderbook-v0',
            env_type=None,
            epsilon=0.090979,
            flat_reward=0.029396,
            gain_delay=30,
            gain_per_step=1.0,
            gamestate=None,
            log_interval=100,
            log_path=None,
            lr=0.00021397,
            max_frames=4,
            max_steps=40,
            nsteps=1,
            num_env=1,
            num_timesteps=40.0,
            play=False,
            reward_ratio=0.29914,
            reward_scale=1.0,
            save_model=True,
            save_path=None,
            save_video_interval=0,
            save_video_length=200,
            seed=None,
        )

        self.max_frames = env_args.max_frames
        self.frames = deque(maxlen=env_args.max_frames)
        self.env = build_env(env_args)

        env_args.env = self.env

        load_model = tf.keras.models.load_model(self.saved_model)
        load_model.summary()
        self.model = Model(
            network=load_model,
            total_timesteps=env_args.num_timesteps,
            **env_args.__dict__
        )

    def emit_prediction(self, data):
        frame = np.asarray(json.loads(data))
        # alog.info(AsciiImage(np.copy(frame), new_width=21))

        self.frames.append(frame)

        if len(self.frames) < self.frames.maxlen:
            return

        obs = tf.constant(np.asarray([self.frames]))
        actions, values, states, _ = self.model.step(obs)
        actions = actions._numpy()

        position = [
            position for position in Positions
            if position.value == actions[0]
        ][0]

        # alog.info((position, actions[0]))

        self.publish(self.job_name, json.dumps({'data': position.value}))

    def run(self):
        self.sub([self.orderbook_channel])


@click.command()
@click.option('--depth', '-d', default=21, type=int)
@click.option('--run-name', '-r', default='default', type=str)
@click.argument('symbol', type=click.Choice(BitmexChannels.__members__))
def main(symbol, **kwargs):

    emitter = PredictionEmitter(
        symbol=BitmexChannels[symbol],
        **kwargs)
    emitter.run()


if __name__ == '__main__':
    main()