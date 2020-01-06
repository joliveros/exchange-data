#!/usr/bin/env python

from exchange_data.tfrecord.dataset import dataset
from exchange_data.tfrecord.tfrecord_directory_info import TFRecordDirectoryInfo
from exchange_data.trading import Positions
from exchange_data.utils import DateTimeUtils
from pytimeparse.timeparse import timeparse
from tgym.envs import OrderBookTradingEnv

import alog
import click
import numpy as np
import random


class TFOrderBookEnv(TFRecordDirectoryInfo, OrderBookTradingEnv):
    def __init__(self, **kwargs):
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

        self.dataset = dataset(batch_size=1, **kwargs)
        self._expected_position = None

    @property
    def position(self):
        return self.last_position

    @position.setter
    def position(self, action: np.float):
        self.last_position = [p for p in Positions if p.value == action][0]

    @property
    def expected_position(self):
        return self._expected_position

    @expected_position.setter
    def expected_position(self, action: np.float):
        self._expected_position = [p for p in Positions if p.value == action][0]

    def _get_observation(self):
        for data in self.dataset:
            timestamp = DateTimeUtils\
                .parse_datetime_str(data.get('datetime').numpy()[-1])

            best_ask = data.get('best_ask').numpy()[-1][-1]
            best_bid = data.get('best_bid').numpy()[-1][-1]
            expected_position = data.get('expected_position').numpy()[-1][-1]
            frame = data.get('frame').numpy()[-1]

            yield timestamp, best_ask, best_bid, expected_position, frame

    def get_observation(self):
        timestamp, best_ask, best_bid, expected_position, frame = next(
            self._get_observation())

        self.frames.append(frame)
        self.expected_position = expected_position
        self.position_history.append(self.position.name[0])

        self.last_datetime = str(timestamp)
        self._last_datetime = timestamp

        self.last_observation = np.copy(self.frames[-1])

        return self.last_observation

    def step(self, action):
        assert self.action_space.contains(action)

        self.position = action

        self.step_count += 1

        if self.step_count >= self.max_episode_length:
            self.done = True

        observation = self.get_observation()

        if self.expected_position == self.position:
            self.reward += 1.0

        reward = self.reset_reward()

        self.print_summary()

        return observation, reward, self.done, {}

    def summary(self):
        summary_keys = [
            'step_count',
            'total_reward'
        ]

        summary = {key: self.__dict__[key] for key in
                   summary_keys}

        summary['position_history'] = \
            ''.join(self.position_history[-1 * self.max_summary:])

        return summary


@click.command()
@click.option('--test-span', default='5m')
@click.option('--summary-interval', '-s', default=120, type=int)
def main(test_span, **kwargs):
    env = TFOrderBookEnv(
        directory_name='default',
        print_ascii_chart=True,
        **kwargs
    )

    for t in range(10):
        env.reset()
        for i in range(timeparse(test_span)):
            env.step(random.randint(0, 2))

    alog.info(alog.pformat(env.summary()))


if __name__ == '__main__':
    main()
