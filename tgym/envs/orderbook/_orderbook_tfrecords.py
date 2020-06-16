#!/usr/bin/env python
from gym.spaces import Discrete
from pandas import DataFrame

from exchange_data.tfrecord.dataset import dataset
from exchange_data.tfrecord.tfrecord_directory_info import TFRecordDirectoryInfo
from exchange_data.trading import Positions
from exchange_data.utils import DateTimeUtils
from pytimeparse.timeparse import timeparse
from tgym.envs.orderbook import OrderBookTradingEnv

import tensorflow_datasets as tfds
import alog
import click
import numpy as np
import random
import tensorflow as tf
import pandas as pd
from optuna import TrialPruned

from tgym.envs.orderbook.ascii_image import AsciiImage


class TFOrderBookEnv(TFRecordDirectoryInfo, OrderBookTradingEnv):
    def __init__(self, trial=None, min_steps=20, max_steps=30, num_env=1,
                 min_change=2.0,
                 **kwargs):
        now = DateTimeUtils.now()
        start_date = kwargs.get('start_date', now)
        end_date = kwargs.get('end_date', now)

        if 'start_date' in kwargs:
            del kwargs['start_date']

        if 'end_date' in kwargs:
            del kwargs['end_date']

        super().__init__(
            min_change=min_change,
            action_space=Discrete(2),
            start_date=start_date,
            end_date=end_date,
            **kwargs
        )
        self.trial = trial
        self.num_env = num_env
        self.max_steps = max_steps
        kwargs['batch_size'] = 1
        self.dataset = dataset(epochs=1, **kwargs)

        frames = []

        for data in tfds.as_numpy(dataset(**kwargs)):
            data['datetime'] = DateTimeUtils\
                .parse_datetime_str(data['datetime'][0].decode('utf8'))

            frames.append(data)

        frames_df = DataFrame.from_dict(frames)
        alog.info(frames_df)
        alog.info(frames_df.shape)

        frames_df.drop_duplicates(subset=['datetime'], inplace=True)

        alog.info(frames_df)
        alog.info(frames_df.shape)

        frames_df = frames_df.set_index('datetime')

        frames_df = frames_df.sort_index()

        frames_df.reset_index(drop=False, inplace=True)

        alog.info(frames_df)
        alog.info(frames_df.shape)

        self.frames_df = frames_df
        self.observations = None
        self.prune_capital = 1.01
        self.min_steps = min_steps
        self.total_steps = 0

    @property
    def best_bid(self):
        return self._best_bid

    @property
    def best_ask(self):
        return self._best_ask

    def _get_observation(self):
        while True:
            for i in range(len(self.frames_df)):
                row = self.frames_df.loc[i]
                best_ask = row['best_ask'][0][0]
                best_bid = row['best_bid'][0][0]
                frame = row['frame'][0]
                timestamp = row['datetime'].to_pydatetime()

                yield timestamp, best_ask, best_bid, frame

    def get_observation(self):
        if self.observations is None:
            self.observations = self._get_observation()

        timestamp, best_ask, best_bid, frame = next(self.observations)

        self._best_ask = best_ask
        self._best_bid = best_bid

        self.frames.append(frame)
        self.position_history.append(self.position.name[0])

        self.last_datetime = str(timestamp)
        self._last_datetime = timestamp

        self.last_observation = np.copy(self.frames)

        return self.last_observation

    def step(self, action):
        assert self.action_space.contains(action)

        self.step_position(action)

        self.reward += self.current_trade.reward

        self.step_count += 1

        if self.trial:
            self.trial.report(self.capital, self.step_count)

        if self.capital < self.min_capital and not self.eval_mode:
            self.done = True
            if self.trial:
                raise TrialPruned()

        if self.step_count >= self.max_steps:
            self.done = True

        # if self.current_trade.pnl <= self.max_negative_pnl:
        #     self.done = True
        #
        #     if self.total_steps >= self.max_negative_pnl_delay and \
        #         self.max_negative_pnl_delay > 0:
        #         raise TrialPruned()

        observation = self.get_observation()

        reward = self.reset_reward()

        self.print_summary()

        return observation, reward, self.done, {}


@click.command()
@click.option('--test-span', default='5m')
@click.option('--summary-interval', '-s', default=120, type=int)
def main(test_span, **kwargs):
    env = TFOrderBookEnv(
        directory_name='default',
        print_ascii_chart=True,
        **kwargs
    )

    for t in range(1):
        env.reset()
        for i in range(timeparse(test_span)):
            env.step(random.randint(0, 1))


if __name__ == '__main__':
    main()
