from abc import ABC
from collections import deque
from datetime import timedelta, datetime

from pyee import EventEmitter

from exchange_data import settings
from exchange_data.streamers._bitmex import BitmexStreamer, OutOfFramesException
from gym import Env
from gym.spaces import Discrete, Box
from pandas import DataFrame
from pytimeparse.timeparse import timeparse

from exchange_data.utils import DateTimeUtils
from tgym.envs.orderbook.utils import Positions

import alog
import click
import logging
import numpy as np


class AlreadyFlatException(Exception):
    pass


class Trade(object):
    def __init__(self, type: str, pnl: float, entry: float, position_exit: float):
        self.type = type
        self.pnl = pnl
        self.entry = entry
        self.position_exit = position_exit

    def __repr__(self):
        return f'{self.type}/{self.pnl}/{self.entry}/{self.position_exit}'


class OrderBookTradingEnv(BitmexStreamer, Env, ABC):
    """
    Orderbook based trading environment.
    """
    window_size: str

    def __init__(
        self,
        trading_fee=0.0/100.0,
        max_loss=-0.01/100.0,
        max_frames=24,
        random_start_date=True,
        orderbook_depth=21,
        window_size='1m',
        sample_interval='1s',
        max_summary=10,
        volatile_ranges=None,
        use_volatile_ranges=False,
        min_std_dev=0.1,
        should_penalize_even_trade=True,
        step_reward=0.000002,
        capital=1.0,
        action_space=None,
        **kwargs
    ):
        kwargs['orderbook_depth'] = orderbook_depth
        kwargs['window_size'] = window_size
        kwargs['sample_interval'] = sample_interval
        kwargs['random_start_date'] = random_start_date
        kwargs['channel_name'] = \
            f'XBTUSD_OrderBookFrame_depth_{orderbook_depth}'

        self._args = locals()
        del self._args['self']

        Env.__init__(self)
        EventEmitter.__init__(self)
        BitmexStreamer.__init__(self, **kwargs)

        self.step_reward = step_reward
        self._best_ask = None
        self._best_bid = None
        self.action_space = Discrete(3) if action_space is None \
            else action_space
        self.ask_diff = 0.0
        self.bid_diff = 0.0
        self.capital = capital
        self.closed_plot = False
        self.entry_price = 0.0
        self.exit_price = 0
        self.frames = deque(maxlen=max_frames)
        self.initial_capital = capital
        self.last_best_ask = None
        self.last_best_bid = None
        self.last_datetime = None
        self.last_index = None
        self.last_observation = np.array([])
        self.last_orderbook = None
        self.last_price_diff = 0.0
        self.last_timestamp = 0
        self.max_frames = max_frames
        self.max_position_duration = (timeparse(self.window_size_str) * 120) \
                                     - max_frames
        self.max_summary = max_summary
        self.min_capital = capital * (1 + max_loss)
        self.min_std_dev = min_std_dev
        self.negative_position_reward_factor = 10**5
        self.position_history = []
        self.reward = 0
        self.should_penalize_even_trade = should_penalize_even_trade
        self.step_count = 0
        self.total_pnl = 0.0
        self.total_reward = 0
        self.trades = []
        self.pnl = 0.0
        self.trading_fee = trading_fee
        self.use_volatile_ranges = use_volatile_ranges
        self.last_position = Positions.Flat
        self.last_bid_side = np.zeros((self.orderbook_depth,))
        self.last_ask_side = np.copy(self.last_bid_side)
        self.position = Positions.Flat
        self.short_pnl = 0.0
        self.long_pnl = 0.0

        if volatile_ranges is None and use_volatile_ranges:
            self.volatile_ranges = self.get_volatile_ranges()
            self._args['volatile_ranges'] = self.volatile_ranges
        else:
            self.volatile_ranges = volatile_ranges

        if use_volatile_ranges:
            nearest_volatile_range = self.volatile_ranges.index\
                .get_loc(self.start_date.timestamp() * (10**3), method='nearest')

            nearest_volatile_range_start = self.volatile_ranges\
                .iloc[nearest_volatile_range].name

            self.start_date = \
                self.parse_db_timestamp(nearest_volatile_range_start)
            self.end_date = self.start_date + timedelta(seconds=self.window_size)

        high = np.full(
            (self.max_frames * (5 + 2 * self.orderbook_depth), ),
            np.inf
        )

        self.observation_space = Box(-high, high, dtype=np.float32)

    def set_position(self, action: np.float):
        if self.should_change_position(action):
            self.change_position(action)

        self.last_position = self.position
        self.short_pnl = self._short_pnl()
        self.long_pnl = self._long_pnl()

        if settings.LOG_LEVEL == logging.DEBUG:
            alog.info(alog.pformat(self.summary()))

    def get_volatile_ranges(self):
        query = f'SELECT bbd FROM (SELECT STDDEV(best_bid) as bbd ' \
            f'from {self.channel_name} GROUP BY time({self.window_size_str})) '\
            f'WHERE bbd > {self.min_std_dev};'

        ranges = self.query(query).get_points(self.channel_name)

        return DataFrame(ranges).set_index('time')

    def reset(self, **kwargs):
        if settings.LOG_LEVEL == logging.DEBUG:
            if self.step_count > 0:
                alog.debug('##### reset ######')
                alog.info(alog.pformat(self.summary()))

        _kwargs = self._args['kwargs']
        del self._args['kwargs']
        _kwargs = {**self._args, **_kwargs, **kwargs}
        new_instance = OrderBookTradingEnv(**_kwargs)
        self.__dict__ = new_instance.__dict__

        # for i in range(self.max_frames):
        #     self.get_observation()

        try:
            for i in range(self.max_frames):
                self.get_observation()
        except (OutOfFramesException, TypeError):
            if not self.random_start_date:
                self._set_next_window()
                kwargs = dict(
                    start_date=self.start_date,
                    end_date=self.end_date
                )
            return self.reset(**kwargs)

        return self.last_observation

    def step(self, action):
        if action > 1:
            raise Exception()

        self.set_position(action)

        assert self.action_space.contains(action)

        done = False

        if self.capital < self.min_capital:
            done = True

        if self.out_of_frames_counter > 30 and not done:
            done = True

        self.step_count += 1

        if self.step_count >= self.max_position_duration:
            done = True

        try:
            observation = self.get_observation()
        except (OutOfFramesException, TypeError):
            observation = self.last_observation
            done = True

        reward = self.reset_reward()
        summary = self.summary()

        if settings.LOG_LEVEL == logging.DEBUG:
            alog.debug(alog.pformat(summary))

        return observation, reward, done, summary

    @property
    def best_bid(self):
        self._best_bid = self.last_orderbook[1][0][0]
        return self._best_bid

    @property
    def position_data(self):
        data_keys = [
            'ask_diff',
            'bid_diff',
            'short_pnl',
            'long_pnl'
        ]

        data = {key: self.__dict__[key] for key in
                   data_keys}

        data['position'] = self.position.value

        return np.array(list(data.values()))

    @property
    def best_ask(self):
        self._best_ask = self.last_orderbook[0][0][0]
        return self._best_ask

    def get_observation(self):
        if self.last_observation.shape[0] > 0:
            self.last_best_ask = self.best_ask
            self.last_best_bid = self.best_bid

        time, orderbook = self._get_observation()

        self.position_history.append(self.position.name[0])
        self.last_datetime = str(time)

        self.last_orderbook = orderbook = np.array(orderbook)

        if orderbook.shape[2] != self.orderbook_depth:
            raise Exception('Orderbook incomplete.')

        if self.last_observation.shape[0] > 0:
            self.bid_diff = self.best_bid - self.last_best_bid
            self.ask_diff = self.best_ask - self.last_best_ask

        position_data = self.position_data

        bid_side = orderbook[1][1] * -1
        ask_side = orderbook[0][1]

        # level_diff = np.stack((self.last_ask_side - ask_side,
        #                        self.last_bid_side - bid_side))
        #
        # self.last_bid_side = bid_side
        # self.last_ask_side = ask_side

        levels = np.stack((ask_side, bid_side))
        frame = np.concatenate((position_data,
                                levels.flatten()))

        self.frames.append(frame)

        self.last_observation = np.concatenate(self.frames)

        return self.last_observation

    def _get_observation(self):
        return next(self)

    def reset_reward(self):
        reward = self.reward
        reward += self.step_reward
        self.total_reward += reward
        self.reward = 0
        return reward

    def close_short(self):
        if self.position.value != Positions.Short.value:
            raise Exception('Not short.')

        pnl = self.short_pnl

        self.trades.append(Trade(
            self.position.name[0],
            pnl, self.entry_price,
            self.best_ask
        ))

        self.total_pnl += pnl
        self.capital += pnl
        self.reward += pnl * 2 if pnl < 0.0 else pnl
        self.entry_price = 0.0

    def close_long(self):
        if self.position != Positions.Long:
            raise Exception('Not long.')

        pnl = self.long_pnl

        trade = Trade(self.position.name[0], pnl, self.entry_price,
                      self.best_bid)

        self.trades.append(trade)

        self.total_pnl += pnl
        self.capital += pnl
        self.reward += pnl * 2 if pnl < 0.0 else pnl
        self.entry_price = 0.0

    def _short_pnl(self):
        if self.entry_price > 0.0:
            pnl = self._pnl(self.best_ask)
        else:
            pnl = 0.0
        return pnl

    def _long_pnl(self):
        if self.entry_price > 0.0:
            pnl = self._pnl(self.best_bid)
        else:
            pnl = 0.0

        return pnl

    def _pnl(self, exit_price):
        diff = 0.0
        if self.last_position.value == Positions.Long.value:
            diff = exit_price - self.entry_price
        elif self.last_position.value == Positions.Short.value:
            diff = self.entry_price - exit_price

        if self.entry_price == 0.0:
            change = 0.0
        else:
            change = diff / self.entry_price

        pnl = (self.capital * change) + (-1 * self.capital * self.trading_fee)

        return pnl

    @property
    def is_short(self):
        return self.position.value == Positions.Short.value

    @property
    def is_long(self):
        return self.position.value == Positions.Long.value

    @property
    def is_flat(self):
        return self.position.value == Positions.Flat.value

    def should_change_position(self, action):
        return self.position.value != action

    def long(self):
        if self.is_long:
            raise Exception('Already long.')
        if self.is_short:
            self.close_short()

        self.position = Positions.Long
        # alog.info(f'set long entry price {self.best_ask}')
        self.entry_price = self.best_ask

    def short(self):
        if self.is_short:
            raise Exception('Already short.')
        if self.is_long:
            self.close_long()

        self.position = Positions.Short
        # alog.info(f'set short entry price {self.best_bid}')
        self.entry_price = self.best_bid

    def change_position(self, action):
        if action == Positions.Long.value:
            self.long()
        elif action == Positions.Short.value:
            self.short()
        elif action == Positions.Flat.value:
            self.flat()

    def flat(self):
        if self.position.value == Positions.Flat.value:
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
            'short_pnl',
            'long_pnl',
            'capital',
            'last_datetime',
            'step_count',
            'total_pnl',
            'total_reward'
        ]

        summary = {key: self.__dict__[key] for key in
                   summary_keys}

        summary['position_history'] = \
            ''.join(self.position_history[-1 * self.max_summary:])
        summary['trades'] = self.trades[-1 * self.max_summary:]

        return summary


@click.command()
@click.option('--test-span', default='2m')
def main(test_span, **kwargs):
    env = OrderBookTradingEnv(
        random_start_date=True,
        use_volatile_ranges=True,
        window_size='1m',
        max_frames=5,
        **kwargs
    )

    env.reset()

    for i in range(timeparse(test_span) - 10):
        env.step(Positions.Long.value)
        # alog.info(alog.pformat(env.summary()))
        # if env.step_count % 5 == 0:
        #     alog.info(env.best_bid)

    env.step(Positions.Flat.value)

    alog.info(alog.pformat(env.summary()))


if __name__ == '__main__':
    main()
