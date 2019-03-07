from abc import ABC
from collections import deque
from datetime import datetime
from time import sleep

import click
from dateutil.tz import tz
from pytimeparse.timeparse import timeparse

from exchange_data.streamers._bitmex import BitmexStreamer
from gym.spaces import Discrete
from tgym.core import Env
from tgym.envs.orderbook import Actions, Positions

import alog
import numpy as np


class AlreadyFlatException(Exception):
    pass


class OrderBookTradingEnv(Env, BitmexStreamer, ABC):
    """
    Orderbook based trading environment.
    """

    def __init__(
        self,
        episode_length=1000,
        trading_fee=1,
        time_fee=0,
        max_frames='2m',
        **kwargs
    ):

        self._args = locals()
        del self._args['self']

        Env.__init__(self)
        BitmexStreamer.__init__(self, **kwargs)

        self.last_timestamp = 0
        self.last_datetime = None
        self._position_pnl = 0
        self.action_space = Discrete(3)
        self.closed_plot = False
        self.entry_price = 0
        self.episode_length = episode_length
        self.exit_price = 0
        self.max_frames = timeparse(max_frames)
        self.frames = deque(maxlen=self.max_frames)
        self.step_count = 0
        self.last_index = None
        self.last_orderbook = None
        self.negative_position_reward_factor = 1.0
        self.position = Positions.Flat
        self.reward = 0
        self.time_fee = time_fee
        self.total_pnl = 0
        self.total_reward = 0
        self.trading_fee = trading_fee
        self.max_position_pnl = 0.0
        self.max_negative_pnl_factor = -0.01
        self.max_position_duration = timeparse('30m')

        for i in range(self.max_frames):
            self.get_observation

        self.observation_space = self.last_observation.shape

    @property
    def max_negative_pnl(self):
        return self.best_bid * self.max_negative_pnl_factor

    @property
    def position_pnl(self):
        if self.is_flat:
            self._position_pnl = 0.0
        elif self.is_long:
            self._position_pnl = self.long_pnl
        elif self.is_short:
            self._position_pnl = self.short_pnl

        return self._position_pnl

    def reset(self):
        kwargs = self._args['kwargs']
        del self._args['kwargs']
        kwargs = {**self._args, **kwargs}
        new_instance = OrderBookTradingEnv(**kwargs)
        self.__dict__ = new_instance.__dict__

        return self.last_observation

    def step(self, action):
        self.reset_reward()
        assert self.action_space.contains(action)

        done = False

        self.reward = -self.time_fee

        if self.should_change_position(action):
            self.change_position(action)

        position_pnl = self.position_pnl

        if position_pnl > self.max_position_pnl:
            self.max_position_pnl = position_pnl

        max_pos_diff = self.max_position_pnl - position_pnl

        self.reward -= max_pos_diff * self.negative_position_reward_factor

        if position_pnl > 0:
            self.reward += position_pnl

        if self.total_pnl + self.position_pnl <= self.max_negative_pnl:
            done = True

        if self.total_reward <= self.max_position_duration * -1:
            done = True

        if done:
            alog.info('Session is complete.')
        else:
            self.step_count += 1

        observation = self.get_observation

        return observation, self.reward, done, self.summary()

    def charge_trading_fee(self):
        self.reward -= self.trading_fee

    @property
    def best_bid(self):
        self._best_bid = self.last_orderbook[0][0][0]
        return self._best_bid

    @property
    def best_ask(self):
        self._best_ask = self.last_orderbook[1][0][0]
        return self._best_ask

    @property
    def last_frame(self):
        return self.frames[-1]

    @property
    def position_data(self):
        data_keys = [
            'total_pnl',
            '_position_pnl',
            'max_position_pnl'
        ]

        data = {key: self.__dict__[key] for key in
                   data_keys}

        return np.array(list(data.values()))

    def local_fromtimestamp(self, value):
        return datetime.fromtimestamp(value/10**9, tz=tz.tzlocal())

    @property
    def get_observation(self):
        time, index, orderbook = next(self)

        self.last_timestamp = time
        self.last_datetime = str(self.local_fromtimestamp(time))

        self.last_index = index
        self.last_orderbook = orderbook = np.array(orderbook)

        index = np.array([index - self.best_ask, index - self.best_bid])
        position_data = self.position_data
        position_data = np.concatenate((position_data, index))

        asks = np.concatenate((orderbook[0]))
        bids = np.concatenate((orderbook[0]))
        resized_position_data = np.zeros((bids.shape[0],))
        resized_position_data[:position_data.shape[0]] = position_data

        frame = np.stack((resized_position_data, asks, bids))

        self.frames.append(frame)

        self.last_observation = np.concatenate(self.frames)

        return self.last_observation

    @staticmethod
    def random_action_fun():
        """The default random action for exploration.
        We hold 80% of the time and buy or sell 10% of the time each.

        Returns:
            numpy.array: array with a 1 on the action index, 0 elsewhere.
        """
        return np.random.multinomial(1, [0.8, 0.1, 0.1])

    def render(self):
        pass

    def reset_reward(self):
        self.total_reward += self.reward
        self.reward = 0

    def close_short(self):
        if self.position != Positions.Short:
            raise Exception('Not short.')

        self.total_pnl += self.short_pnl
        self.reward += self.total_pnl

    def close_long(self):
        if self.position != Positions.Long:
            raise Exception('Not long.')

        self.total_pnl += self.long_pnl
        self.reward += self.total_pnl

    @property
    def short_pnl(self):
        return self.entry_price - self.best_bid

    @property
    def long_pnl(self):
        return self.best_ask - self.entry_price

    @property
    def is_short(self):
        return self.position == Positions.Short

    @property
    def is_long(self):
        return self.position == Positions.Long

    @property
    def is_flat(self):
        return self.position == Positions.Flat

    def should_change_position(self, action):
        return self.position.value != action

    def long(self):
        if self.is_long:
            raise Exception('Already long.')
        self.charge_trading_fee()
        if self.is_short:
            self.close_short()

        self.position = Positions.Long
        self.entry_price = self.best_bid

    def short(self):
        if self.is_short:
            raise Exception('Already short.')
        self.charge_trading_fee()
        if self.is_long:
            self.close_long()

        self.position = Positions.Short
        self.entry_price = self.best_ask

    def change_position(self, action):
        if action == Positions.Long.value:
            self.long()
        elif action == Positions.Short.value:
            self.short()
        elif action == Positions.Flat.value:
            self.flat()

    def flat(self):
        if self.position == Positions.Flat:
            raise AlreadyFlatException()

        if self.is_long:
            self.close_long()
        elif self.is_short:
            self.close_short()

        self.position = Positions.Flat

    def summary(self):
        summary_keys = [
            '_best_ask',
            '_best_bid',
            '_position_pnl',
            'last_datetime',
            'max_position_pnl',
            'position',
            'total_pnl',
            'total_reward',
            'step_count'
        ]

        summary = {key: self.__dict__[key] for key in
                   summary_keys}

        alog.info(alog.pformat(summary))
        return summary


@click.command()
@click.option('--test-span', default='1m')
def main(test_span, **kwargs):
    env = OrderBookTradingEnv(
        window_size='30s',
        random_start_date=True,
        **kwargs
    )

    for i in range(timeparse(test_span)):
        env.step(Positions.Long.value)
        sleep(0.2)

    env.step(Positions.Flat.value)


if __name__ == '__main__':
    main()
