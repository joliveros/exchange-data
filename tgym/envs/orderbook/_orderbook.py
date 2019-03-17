import sys
import traceback
from abc import ABC
from collections import deque
from datetime import datetime
from dateutil.tz import tz
from gym.spaces import Discrete, Box

from exchange_data.streamers._bitmex import BitmexStreamer, OutOfFramesException
from gym import Env
from pytimeparse.timeparse import timeparse
from time import sleep
import alog
import click
import numpy as np

from tgym.envs.orderbook.utils import Positions


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


class OrderBookTradingEnv(Env, BitmexStreamer, ABC):
    """
    Orderbook based trading environment.
    """

    def __init__(
        self,
        episode_length=1000,
        trading_fee=0.075/100.0,
        max_loss=-2/100.0,
        no_pnl_reward=-1.0,
        max_frames=100,
        random_start_date=True,
        orderbook_depth=21,
        window_size='10m',
        sample_interval='1s',
        max_summary=50,
        should_penalize_even_trade=True,
        capital=1.0,
        **kwargs
    ):
        kwargs['random_start_date'] = random_start_date
        kwargs['orderbook_depth'] = orderbook_depth
        kwargs['window_size'] = window_size
        kwargs['sample_interval'] = sample_interval
        kwargs['channel_name'] = \
            f'XBTUSD_OrderBookFrame_depth_{orderbook_depth}'

        self._args = locals()
        del self._args['self']

        Env.__init__(self)
        BitmexStreamer.__init__(self, **kwargs)
        self.initial_capital = capital
        self.min_capital = capital * (1 + max_loss)
        self.capital = self.initial_capital
        self.should_penalize_even_trade = should_penalize_even_trade
        self.max_summary = max_summary
        self.no_pnl_reward = no_pnl_reward
        self.last_timestamp = 0
        self.last_datetime = None
        self.action_space = Discrete(3)
        self.closed_plot = False
        self.entry_price = 0.0
        self.episode_length = episode_length
        self.exit_price = 0
        self.max_frames = max_frames
        self.frames = deque(maxlen=self.max_frames)
        self.step_count = 0
        self.last_index = None
        self.last_orderbook = None
        self.negative_position_reward_factor = 1.0
        self.position = Positions.Flat
        self.reward = 0
        self.total_pnl = 0.0
        self.total_reward = 0
        self.trading_fee = trading_fee
        self.max_position_duration = 500
        self.position_history = []
        self.bid_diff = 0.0
        self.ask_diff = 0.0
        self.last_observation = np.array([])
        self.last_best_ask = None
        self.last_best_bid = None
        self._best_bid = None
        self._best_ask = None
        self.trades = []
        self.last_price_diff = 0.0

        high = np.full(
            (self.max_frames * (4 + 4 * self.orderbook_depth), ),
            np.inf
        )

        self.observation_space = Box(-high, high, dtype=np.float32)

    def reset(self, **kwargs):
        alog.debug('##### reset ######')

        if self.step_count > 0:
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
        # sleep(1)

        assert self.action_space.contains(action)

        done = False

        if self.should_change_position(action):
            self.change_position(action)

        if self.capital < self.min_capital:
            done = True

        if self.out_of_frames_counter > 30 and not done:
            done = True

        self.step_count += 1

        try:
            observation = self.get_observation()
        except (OutOfFramesException, TypeError):
            observation = self.last_observation
            done = True

        reward = self.reset_reward()

        # alog.info(alog.pformat(self.summary()))
        return observation, reward, done, self.summary()

    @property
    def best_bid(self):
        self._best_bid = self.last_orderbook[0][0][0]
        return self._best_bid

    @property
    def best_ask(self):
        self._best_ask = self.last_orderbook[1][0][0]
        return self._best_ask

    @property
    def position_data(self):
        data_keys = [
            'ask_diff',
            'bid_diff',
            'total_pnl'
        ]

        data = {key: self.__dict__[key] for key in
                   data_keys}

        data['position'] = self.position.value

        return np.array(list(data.values()))

    def get_observation(self):
        if self.last_observation.shape[0] > 0:
            self.last_best_ask = self.best_ask
            self.last_best_bid = self.best_bid

        time, orderbook = self._get_observation()
        self.position_history.append(self.position.name[0])
        self.last_datetime = str(time)

        self.last_orderbook = orderbook = np.array(orderbook)

        if self.last_observation.shape[0] > 0:
            self.bid_diff = self.best_bid - self.last_best_bid
            self.ask_diff = self.best_ask - self.last_best_ask

        position_data = self.position_data

        frame = np.concatenate((position_data, orderbook.flatten()))
        self.frames.append(frame)

        self.last_observation = np.concatenate(self.frames)

        return self.last_observation

    def _get_observation(self):
        return next(self)

    def reset_reward(self):
        reward = self.reward
        self.total_reward += reward
        self.reward = 0
        return reward

    def close_short(self):
        if self.position != Positions.Short:
            raise Exception('Not short.')

        pnl = self.short_pnl

        self.trades.append(Trade(
            self.position.name[0],
            pnl, self.entry_price,
            self.best_bid
        ))

        self.total_pnl += pnl
        self.capital += pnl
        self.reward += pnl
        self.entry_price = 0.0

    def close_long(self):
        if self.position != Positions.Long:
            raise Exception('Not long.')

        pnl = self.long_pnl

        self.trades.append(Trade(
            self.position.name[0],
            pnl,
            self.entry_price,
            self.best_ask
        ))

        self.total_pnl += pnl
        self.capital += pnl
        self.reward += pnl
        self.entry_price = 0.0

    @property
    def short_pnl(self):
        if self.entry_price > 0.0:
            pnl = self.pnl(self.best_bid)
        else:
            pnl = 0.0
        return pnl

    def pnl(self, exit_price):
        # alog.info(self.entry_price)
        # alog.info(exit_price)
        diff = self.entry_price - exit_price
        change = diff / self.entry_price
        pnl = (self.capital * change) + (-1 * self.capital * self.trading_fee)
        # alog.info(pnl)
        return pnl

    @property
    def long_pnl(self):
        if self.entry_price > 0.0:
            pnl = self.pnl(self.best_ask)
        else:
            pnl = 0.0
        return pnl

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
@click.option('--test-span', default='10m')
def main(test_span, **kwargs):
    env = OrderBookTradingEnv(
        random_start_date=True,
        **kwargs
    )

    env.reset()

    for i in range(timeparse(test_span) - 101):
        env.step(Positions.Long.value)
        # sleep(1)

    env.step(Positions.Flat.value)
    alog.info(alog.pformat(env.summary()))


if __name__ == '__main__':
    main()
