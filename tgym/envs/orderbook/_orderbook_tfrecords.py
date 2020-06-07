#!/usr/bin/env python
from gym.spaces import Discrete

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
import optuna
import tensorflow as tf

from tgym.envs.orderbook.ascii_image import AsciiImage


class TFOrderBookEnv(TFRecordDirectoryInfo, OrderBookTradingEnv):
    def __init__(self, trial, min_steps, max_steps=30, num_env=1,
                 min_change=2.0,
                 **kwargs):
        now = DateTimeUtils.now()
        start_date = kwargs.get('start_date', now)
        end_date = kwargs.get('end_date', now)
        max_loss = -0.5/100.0

        if 'start_date' in kwargs:
            del kwargs['start_date']

        if 'end_date' in kwargs:
            del kwargs['end_date']

        super().__init__(
            max_loss=max_loss,
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
        self.dataset = dataset(**kwargs)
        self._expected_position = None
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
        if self.observations is None:
            self.observations = self._get_observation()

        timestamp, best_ask, best_bid, expected_position, frame = next(self.observations)

        self._best_ask = best_ask
        self._best_bid = best_bid

        # alog.info(AsciiImage(np.copy(frame), new_width=21))

        self.frames.append(frame)
        self.expected_position = expected_position
        self.position_history.append(self.position.name[0])

        self.last_datetime = str(timestamp)
        self._last_datetime = timestamp

        self.last_observation = np.copy(self.frames)

        return self.last_observation

    def step(self, action):
        alog.info(action)

        assert self.action_space.contains(action)

        self.step_position(action)

        self.reward += self.current_trade.reward

        self.step_count += 1

        self.trial.report(self.capital, self.step_count)

        if self.total_steps > self.gain_delay:
            expected_capital = 1.0 + (
                    (self.step_count - self.gain_delay) * self.gain_per_step)

            alog.info(f'### expected capital '
                      f'{(self.capital, expected_capital)} ###')

            if self.capital < expected_capital:
                self.done = True
                raise optuna.TrialPruned()

        min_capital = 1 + self.max_loss
        alog.info((self.capital, min_capital, self.max_loss))

        if self.capital < min_capital:
            self.done = True

        # alog.info((self.step_count, self.min_steps, self.capital,
        #            self.prune_capital))
        # if self.step_count >= self.min_steps and self.capital < self.prune_capital:
        #     self.done = True
        #     alog.info('#### PRUNE ####')
        #     raise optuna.TrialPruned()

        if self.step_count >= self.max_steps:
            self.done = True

        observation = self.get_observation()

        reward = self.reset_reward()

        alog.info(reward)

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
