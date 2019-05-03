import traceback
import json
import random
from abc import ABC
from collections import deque
from datetime import timedelta, datetime

import easy_tf_log
from matplotlib.figure import Figure

from exchange_data import settings
from exchange_data.streamers._bitmex import BitmexStreamer, OutOfFramesException
from exchange_data.utils import DateTimeUtils
from gym import Env
from gym.spaces import Discrete, Box, Dict
from pandas import DataFrame
from pyee import EventEmitter
from pytimeparse.timeparse import timeparse
from tgym.envs.orderbook.ascii_image import AsciiImage
from tgym.envs.orderbook.utils import Positions
from skimage import data, color

import alog
import click
import logging
import numpy as np
import matplotlib.pyplot as plt


class AlreadyFlatException(Exception):
    pass


class Trade(object):
    def __init__(self, type: str, pnl: float, entry: float,
                 position_exit: float,
                 position_length: int):
        self.position_length = position_length
        self.type = type
        self.pnl = pnl
        self.entry = entry
        self.position_exit = position_exit

    def __repr__(self):
        return f'{self.type}/{self.pnl}/{self.entry}/{self.position_exit}/' \
            f'{self.position_length}'


class PositionLengthExceeded(Exception):
    pass


class PlotOrderbook(object):
    def __init__(self, frame_width):
        self.frame_width = frame_width

        if 'fig' not in self.__dict__:
            plt.close()
            fig, frames = plt.subplots(1, 2, figsize=(2, 1), dpi=self.frame_width)

            ax1, ax2 = frames
            self.fig = fig
            self.ax1 = ax1
            self.ax2 = ax2
            # self.ax2 = fig.add_subplot(1, 2, 2, frame_on=False)

    def hide_ticks_and_values(self, frame):
        frame.axis('off')
        # frame.axes.get_xaxis().set_visible(False)
        # frame.axes.get_yaxis().set_visible(False)
        # frame.axes.get_xaxis().set_ticks([])
        # frame.axes.get_yaxis().set_ticks([])

class OrderBookTradingEnv(BitmexStreamer, Env, PlotOrderbook, ABC):
    """
    Orderbook based trading environment.
    """
    window_size: str

    def __init__(
        self,
        logger=None,
        trading_fee=0.0/100.0,
        max_loss=-0.1/100.0,
        random_start_date=True,
        orderbook_depth=21,
        window_size='2m',
        sample_interval='1s',
        max_summary=10,
        max_frames=100,
        volatile_ranges=None,
        use_volatile_ranges=True,
        min_std_dev=2.0,
        should_penalize_even_trade=True,
        step_reward=0.000005,
        capital=1.0,
        action_space=None,
        is_training=True,
        print_ascii_chart=False,
        summary_interval=120,
        min_change=3.0,
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

        self.frame_width = 96

        PlotOrderbook.__init__(self, self.frame_width)

        self.position_pnl_history = np.array([])
        self.position_pnl_diff_history = np.array([])
        self.done = False
        self.min_change = min_change
        self.logger = logger
        self.summary_interval = summary_interval
        self.name = None
        self.max_n_noops = 10
        self.last_spread = 0.0
        self.is_training = is_training
        self.max_frames = max_frames
        self.frames = deque(maxlen=max_frames)
        self.price_frames = deque(maxlen=max_frames)
        self.position_data_history = deque(maxlen=max_frames * 12)
        self.long_pnl_history = []
        self._best_ask = 0.0
        self._best_bid = 0.0
        self.action_space = Discrete(3) if action_space is None \
            else action_space
        self.ask_diff = 0.0
        self.bid_diff = 0.0
        self.capital = capital
        self.closed_plot = False
        self.entry_price = 0.0
        self.exit_price = 0
        self.initial_capital = capital
        self.last_best_ask = None
        self.last_best_bid = None
        self.last_datetime = None
        self.last_index = None
        self.last_orderbook = None
        self.last_price_diff = 0.0
        self.last_timestamp = 0
        self.max_episode_length_str = '10m'
        self.max_episode_length = timeparse(self.max_episode_length_str)
        self.max_summary = max_summary
        self.min_capital = capital * (1 + max_loss)
        self.min_std_dev = min_std_dev
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
        self.print_ascii_chart = print_ascii_chart
        self.position_repeat = 0

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

        self.start_date -= timedelta(seconds=self.max_frames)

        high = np.full(
            (max_frames, self.frame_width, self.frame_width * 2),
            1.0,
            dtype=np.float32
        )
        low = np.full(
            (max_frames, self.frame_width, self.frame_width * 2),
            0.0,
            dtype=np.float32
        )

        self.observation_space = Box(low, high, dtype=np.float32)

        self.last_observation = None

    @staticmethod
    def get_action_meanings():
        return ['NOOP']

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def set_position(self, action: np.float):
        self.last_position = self.position

        if self.should_change_position(action):
            self.change_position(action)
            self.position_repeat = 0

        self.position_repeat += 1

        self.short_pnl = self._short_pnl()
        self.long_pnl = self._long_pnl()

        # if settings.LOG_LEVEL == logging.DEBUG:
        #     alog.info(alog.pformat(self.summary()))

    def get_volatile_ranges(self):
        query = f'SELECT bbd FROM (SELECT STDDEV(best_bid) as bbd ' \
            f'from {self.channel_name} GROUP BY time({self.max_episode_length_str})) '\
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

        n_noops = np.random.randint(low=self.max_frames,
                                    high=self.max_frames + self.max_n_noops + 1)
        # for i in range(n_noops):
        #     self.get_observation()

        try:
            for _ in range(n_noops):
                self.get_observation()
        except (OutOfFramesException, TypeError, Exception):
            if not self.random_start_date:
                self._set_next_window()
            return self.reset(**kwargs)

        return self.last_observation

    def step(self, action):
        if action > 1:
            raise Exception()

        assert self.action_space.contains(action)

        self.set_position(action)

        # if self.capital < self.min_capital:
        #     self.done = True

        self.step_count += 1

        if self.step_count >= self.max_episode_length:
            self.done = True

        # observation = self.get_observation()

        try:
            observation = self.get_observation()
        except (OutOfFramesException, TypeError, Exception):
            observation = self.last_observation
            self.done = True

        reward = self.reset_reward()

        self.print_summary()

        # reward = np.clip(reward, -1, +1)

        return observation, reward, self.done, {}

    def print_summary(self):
        if settings.LOG_LEVEL == logging.DEBUG and not self.is_training:
            if self.step_count % self.summary_interval == 0:
                alog.debug(alog.pformat(self.summary()))

    @property
    def best_bid(self):
        self._best_bid = self.last_orderbook[1][0][0]
        return self._best_bid

    @property
    def position_data(self):
        data_keys = [
            '_best_bid',
            '_best_ask',
            'ask_diff',
            'bid_diff',
            # 'short_pnl',
            # 'long_pnl',
            'last_spread'
        ]

        data = {key: self.__dict__[key] for key in
                   data_keys}

        # data['position'] = self.position.value

        return np.array(list(data.values()))

    @property
    def best_ask(self):
        self._best_ask = self.last_orderbook[0][0][0]
        return self._best_ask

    @property
    def last_orderbook_levels(self):
        return self.last_observation

    @staticmethod
    def normalized(a, axis=-1, order=2):
        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2==0] = 1
        return a / np.expand_dims(l2, axis)

    def get_observation(self):
        if self.last_orderbook_levels is not None:
            self.last_best_ask = self.best_ask
            self.last_best_bid = self.best_bid

        time, orderbook = self._get_observation()
        orderbook[1][1] = np.multiply(orderbook[1][1], -1.0)
        self.position_history.append(self.position.name[0])

        self.last_datetime = str(time)
        self._last_datetime = time

        self.last_orderbook = orderbook = np.array(orderbook)

        bids = np.array([orderbook[0][0], orderbook[0][1]])
        self.bids = bids.reshape((2, bids.shape[1])).swapaxes(0, 1)[:10]
        # alog.info(self.bids[:10])
        asks = np.array([orderbook[1][0], orderbook[1][1]])
        self.asks = asks.reshape((2, asks.shape[1])).swapaxes(0, 1)[:10]
        # alog.info(self.asks[:10])

        self.position_data_history.appendleft(self.position_data)

        self.long_pnl_history.append(self.long_pnl)

        plot = self.plot_orderbook()

        if orderbook.shape[2] != self.orderbook_depth:
            raise Exception('Orderbook incomplete.')

        if self.last_orderbook_levels is not None:
            self.bid_diff = self.best_bid - self.last_best_bid
            self.ask_diff = self.best_ask - self.last_best_ask
            self.last_spread = self.best_ask - self.best_bid

        self.frames.append(plot)

        self.last_observation = np.copy(self.frames)

        return self.last_observation

    def plot_orderbook(self):
        self.ax1.clear()
        ax1 = self.ax1
        fig = self.fig

        bsizeacc = 0
        bhys = []    # bid - horizontal - ys
        bhxmins = [] # bid - horizontal - xmins
        bhxmaxs = [] # ...
        bvxs = []
        bvymins = []
        bvymaxs = []
        asizeacc = 0
        ahys = []
        ahxmins = []
        ahxmaxs = []
        avxs = []
        avymins = []
        avymaxs = []
        orderbook = self.last_orderbook

        bids = self.bids
        asks = self.asks

        for (p1, s1), (p2, s2) in zip(bids, bids[1:]):
            if bsizeacc == 0:
                bvxs.append(p1)
                bvymins.append(0.0)
                bvymaxs.append(s1)
                bsizeacc += s1
            bvymins.append(bsizeacc)
            bhys.append(bsizeacc)
            bhxmins.append(p2)
            bhxmaxs.append(p1)
            bvxs.append(p2)
            bsizeacc += s2
            bvymaxs.append(bsizeacc)

        for (p1, s1), (p2, s2) in zip(asks, asks[1:]):
            if asizeacc == 0:
                avxs.append(p1)
                avymins.append(0.0)
                avymaxs.append(s1)
                asizeacc += s1

            avymins.append(asizeacc)
            ahys.append(asizeacc)
            ahxmins.append(p1)
            ahxmaxs.append(p2)
            avxs.append(p2)
            asizeacc += s2
            avymaxs.append(asizeacc)

        ax1.hlines(bhys, bhxmins, bhxmaxs, color="green")
        ax1.vlines(bvxs, bvymins, bvymaxs, color="green")
        ax1.hlines(ahys, ahxmins, ahxmaxs, color="red")
        ax1.vlines(avxs, avymins, avymaxs, color="red")
        self.plot_price_over_time()
        plt.autoscale(tight=True)
        self.hide_ticks_and_values(ax1)
        fig.patch.set_visible(False)
        fig.canvas.draw()

        img = fig.canvas.renderer._renderer
        img = np.array(img)
        img = color.rgb2gray(img)

        if settings.LOG_LEVEL == logging.DEBUG and self.print_ascii_chart:
            if self.step_count % self.summary_interval == 0:
                alog.info(AsciiImage(img))
                # plt.show()
                # traceback.print_stack()

        return img

    def plot_price_over_time(self):
        self.ax2.clear()
        ax2 = self.ax2
        bid_ask = np.array(self.position_data_history)[:-1, :2]
        bid_ask = np.flip(bid_ask, axis=0)
        bid_ask = bid_ask.swapaxes(1, 0)
        avg = np.add(bid_ask[0], bid_ask[1]) / 2
        ax2.plot(avg, color='orange')
        self.hide_ticks_and_values(ax2)

    def _get_observation(self):

        time = None
        orderbook = None

        while self.last_timestamp == time or time is None:
            time, orderbook = next(self)

        self.last_timestamp = time
        return time, orderbook

    def reset_reward(self):
        reward = self.reward
        self.total_reward += reward
        self.reward = 0.0
        return reward

    def close_short(self):
        if self.position.value != Positions.Short.value:
            raise Exception('Not short.')

        pnl = self.short_pnl

        self.trades.append(Trade(
            self.position.name[0],
            pnl, self.entry_price,
            self.best_ask,
            self.position_repeat
        ))

        self.total_pnl += pnl
        self.capital += pnl
        self.reward += 1.0 if pnl > self.min_profit else -1.0
        self.entry_price = 0.0

    def close_long(self):
        if self.position != Positions.Long:
            raise Exception('Not long.')

        pnl = self.long_pnl

        trade = Trade(self.position.name[0], pnl, self.entry_price,
                      self.best_bid, self.position_repeat)

        self.trades.append(trade)

        self.total_pnl += pnl
        self.capital += pnl
        self.reward += 1.0 if pnl > self.min_profit else -1.0
        self.entry_price = 0.0

    @property
    def min_profit(self):
        if self.best_bid > 0:
            change = self.min_change / self.last_best_bid
            return (self.capital * change) + (-1 * self.capital * self.trading_fee)
        else:
            return 0.0

    def _short_pnl(self):
        if self.entry_price > 0.0:
            pnl = self._pnl(self.best_ask)
        else:
            pnl = 0.0
        return pnl

    def _long_pnl(self):
        if self.entry_price > 0.0:
            best_bid = self.best_bid
        else:
            best_bid = self.entry_price

        pnl = self._pnl(best_bid)

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

    def log_trades(self):
        trade_count = len(self.trades)
        self.logger.logkv('trades/count', trade_count)

        if trade_count > 0:
            self.logger.logkv('trades/mean_pnl',
                              sum([trade.pnl for trade in self.trades]) / trade_count)
            self.logger.logkv(
                'trades/mean_len',
                sum([trade.position_length for trade in self.trades]) / trade_count
            )

@click.command()
@click.option('--test-span', default='5m')
@click.option('--summary-interval', '-s', default=120, type=int)
def main(test_span, **kwargs):
    env = OrderBookTradingEnv(
        random_start_date=True,
        use_volatile_ranges=False,
        window_size='30s',
        is_training=False,
        print_ascii_chart=True,
        **kwargs
    )

    env.reset()

    for i in range(timeparse(test_span) - 10):
        env.step(Positions.Long.value)
        # alog.info(alog.pformat(env.summary()))
        if env.step_count % 5 == 0:
            alog.info(env.best_bid)

    env.step(Positions.Flat.value)

    alog.info(alog.pformat(env.summary()))


if __name__ == '__main__':
    main()
