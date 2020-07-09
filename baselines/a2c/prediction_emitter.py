#!/usr/bin/env python
import distutils
import gzip
import logging
from argparse import Namespace
from collections import deque
from datetime import timedelta
from distutils.util import strtobool

from baselines.a2c.model import Model
from baselines.run import build_env
from exchange_data import settings
from exchange_data.channels import BitmexChannels
from exchange_data.emitters import Messenger
from exchange_data.streamers._orderbook_level import OrderBookLevelStreamer
from exchange_data.trading import Positions
from exchange_data.utils import DateTimeUtils
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


class FrameNormalizer(object):
    observed_max = 0.0

    def normalize_frame(self, orderbook_levels, volume_max):
        orderbook_levels = np.concatenate((orderbook_levels[0],
                                          orderbook_levels[1]))
        orderbook_levels = np.sort(orderbook_levels, axis=0)
        orderbook_levels = np.delete(orderbook_levels, 0, 1)

        max_value = np.max(orderbook_levels)

        if max_value > self.observed_max:
            self.observed_max = max_value

        orderbook_levels = np.reshape(orderbook_levels,
                                      (orderbook_levels.shape[0], 1)) / volume_max
        return np.clip(orderbook_levels, a_min=0.0, a_max=volume_max)

class PredictionEmitter(Messenger, TradeJob, FrameNormalizer):
    def __init__(self, symbol, depth, run_name, **kwargs):
        self.symbol = symbol

        super().__init__(symbol=symbol, decode=False, **kwargs)
        self.orderbook_channel = f'XBTUSD_OrderBookFrame_depth_{depth}_2s'
        self.saved_model = f'{Path.home()}/.exchange-data/models/a2c/model_export/{run_name}.h5'

        self.on(self.orderbook_channel, self.emit_prediction)
        self.on('should_trade', self.should_trade)

        env_args = Namespace(
            alg='a2c',
            batch_size=1,
            directory_name='default',
            env='tf-orderbook-v0',
            env_type=None,
            epsilon=1e-7,
            flat_reward=1.0,
            gain_delay=30,
            gain_per_step=1.0,
            gamestate=None,
            log_interval=100,
            log_path=None,
            lr=0.00021397,
            max_frames=48,
            max_loss=-0.02,
            max_steps=40,
            kernel_dim=4,
            min_steps=50,
            nsteps=7,
            num_env=1,
            num_timesteps=40.0,
            play=False,
            reward_ratio=1.0,
            reward_scale=1.0,
            save_model=True,
            save_path=None,
            save_video_interval=0,
            save_video_length=200,
            should_load_dataset=False,
            seed=None,
            trial={},
        )
        self.trading_enabled = False
        self.max_frames = env_args.max_frames
        self.frames = deque(maxlen=env_args.max_frames)
        self.env = build_env(env_args)

        self.load_previous_frames(depth)

        env_args.env = self.env

        load_model = tf.keras.models.load_model(self.saved_model)
        load_model.summary()
        self.model = Model(
            network=load_model,
            total_timesteps=env_args.num_timesteps,
            **env_args.__dict__
        )

    def should_trade(self, should_trade):
        self.trading_enabled = strtobool(should_trade.decode())

    def load_previous_frames(self, depth):
        now = DateTimeUtils.now()
        start_date = now - timedelta(seconds=48)
        levels = OrderBookLevelStreamer(
            start_date=start_date,
            end_date=DateTimeUtils.now(),
            database_name='bitmex',
            depth=depth,
            groupby='2s',
            window_size='48s',
            sample_interval='48s'
        )

        for timestamp, best_ask, best_bid, orderbook_img in levels:
            if orderbook_img:
                orderbook_img = np.asarray(json.loads(orderbook_img))
                try:
                    orderbook_img = self.normalize_frame(orderbook_img)
                    self.frames.append(orderbook_img)
                except:
                    pass


    def _emit_prediction(self, data):
        frame = json.loads(data)['fields']['data']
        frame = json.loads(frame)

        self.frames.append(self.normalize_frame(frame))

        if len(self.frames) < self.frames.maxlen:
            return

        frames = np.asarray([self.frames])

        obs = tf.constant(frames)
        actions, values, states, _ = self.model.step(obs)
        actions = actions._numpy()

        position = [
            position for position in Positions
            if position.value == actions[0]
        ][0]

        alog.info((position, actions[0]))

        self.publish(self.job_name, json.dumps({'data': position.value}))

    def emit_prediction(self, data):
        if self.trading_enabled:
            try:
                self._emit_prediction(data)
            except:
                pass
        else:
            self.publish(self.job_name, json.dumps({'data': Positions.Flat.value}))


    def run(self):
        self.sub([self.orderbook_channel, 'should_trade'])


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
