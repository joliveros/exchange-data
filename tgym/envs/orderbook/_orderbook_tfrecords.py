#!/usr/bin/env python

from exchange_data.tfrecord.dataset import dataset
from exchange_data.tfrecord.tfrecord_directory_info import TFRecordDirectoryInfo
from exchange_data.trading import Positions
from exchange_data.utils import DateTimeUtils
from pytimeparse.timeparse import timeparse
from tgym.envs.orderbook import OrderBookTradingEnv

import alog
import click
import numpy as np
import random
import tensorflow as tf


class TFOrderBookEnv(TFRecordDirectoryInfo, OrderBookTradingEnv):
    def __init__(self, max_steps=30, **kwargs):
        now = DateTimeUtils.now()
        start_date = kwargs.get('start_date', now)
        end_date = kwargs.get('end_date', now)

        if 'start_date' in kwargs:
            del kwargs['start_date']

        if 'end_date' in kwargs:
            del kwargs['end_date']

        super().__init__(
            start_date=start_date,
            end_date=end_date,
            **kwargs
        )
        self.max_steps = max_steps
        self.dataset = dataset(batch_size=1, **kwargs)
        self._expected_position = None

    @property
    def best_bid(self):
        return self._best_bid

    @property
    def best_ask(self):
        return self._best_ask

    @property
    def expected_position(self):
        return self._expected_position

    @expected_position.setter
    def expected_position(self, action: np.float):
        self._expected_position = [p for p in Positions if p.value == action][0]

    def _get_observation(self):
        for data in self.dataset:
            timestamp = DateTimeUtils.parse_datetime_str(data['datetime'].numpy()[0])

            best_ask = data.get('best_ask').numpy()[-1][-1]
            best_bid = data.get('best_bid').numpy()[-1][-1]
            expected_position = data.get('expected_position').numpy()[-1][-1]
            frame = data.get('frame').numpy()[0]

            yield timestamp, best_ask, best_bid, expected_position, frame

    def get_observation(self):
        timestamp, best_ask, best_bid, expected_position, frame = next(
            self._get_observation())
        self._best_ask = best_ask
        self._best_bid = best_bid

        self.frames.append(frame)
        self.expected_position = expected_position
        self.position_history.append(self.position.name[0])

        self.last_datetime = str(timestamp)
        self._last_datetime = timestamp

        self.last_observation = np.copy(self.frames[-1])

        return self.last_observation

    def step(self, action):
        assert self.action_space.contains(action)

        self.step_position(action)

        self.step_count += 1

        if self.step_count >= self.max_steps:
            self.done = True

        observation = self.get_observation()

        if self.expected_position == self.position:
            self.reward += 1.0
        else:
            self.reward -= 1.0

        reward = self.reset_reward()

        self.print_summary()

        alog.info(alog.pformat(self.summary()))

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
            env.step(random.randint(0, 2))


if __name__ == '__main__':
    main()
