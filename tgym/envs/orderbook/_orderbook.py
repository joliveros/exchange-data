from collections import deque
from datetime import timedelta
from exchange_data import settings
from exchange_data.trading import Positions
from gym import Env
from gym.spaces import Discrete, Box
from pandas import DataFrame
from pytimeparse.timeparse import timeparse

from exchange_data.utils import DateTimeUtils
from tgym.envs.orderbook._plot_orderbook import PlotOrderbook
from tgym.envs.orderbook.utils import Logging
from tgym.envs.orderbook.trade import Trade
from tgym.envs.orderbook.trade.long import LongTrade
from tgym.envs.orderbook.trade.flat import FlatTrade
from tgym.envs.orderbook.trade.short import ShortTrade
from tgym.envs.orderbook.ascii_image import AsciiImage

import alog
import click
import logging
import matplotlib.pyplot as plt
import numpy as np
import random
import traceback


class OrderBookIncompleteException(Exception):
    pass


class OrderBookTradingEnv(Logging, Env):
    """
    Orderbook based trading environment.
    """
    window_size: str

    def __init__(
        self,
        sequence_length=12,
        depth=24,
        min_steps=10,
        levels=30,
        summary_interval=120,
        database_name = 'bitmex',
        logger=None,
        leverage=1.0,
        trading_fee=4e-4,
        max_loss=-5.0/100.0,
        window_size='2m',
        sample_interval='1s',
        max_summary=10,
        max_frames=2,
        volatile_ranges=None,
        use_volatile_ranges=False,
        min_std_dev=2.0,
        should_penalize_even_trade=True,
        step_reward_ratio=1.0,
        step_reward=0.001,
        capital=1.0,
        action_space=None,
        is_training=True,
        print_ascii_chart=False,
        min_change=0.00,
        max_negative_pnl=-0.05,
        frame_width=224,
        reward_ratio=1.0,
        gain_per_step=1.0,
        gain_delay=30,
        is_test=False,
        **kwargs
    ):
        kwargs['database_name'] = database_name
        kwargs['window_size'] = window_size
        kwargs['sample_interval'] = sample_interval

        self._args = locals()
        self.is_test = is_test
        self.last_observation = None
        self.last_timestamp = 0
        self.reset_class = OrderBookTradingEnv

        super().__init__(
            **kwargs
        )

        self.min_steps = min_steps
        self.max_negative_pnl = max_negative_pnl
        self.gain_per_step = gain_per_step
        self.gain_delay = gain_delay
        self._reward_ratio = reward_ratio
        self.step_reward_ratio = step_reward_ratio
        self.step_reward = step_reward
        self.leverage = leverage
        self.capital = capital
        self.frame_width = frame_width
        self.position_pnl_history = np.array([])
        self.position_pnl_diff_history = np.array([])
        self._done = False
        self.min_change = min_change
        self.logger = logger
        self.summary_interval = summary_interval
        self.max_n_noops = 10
        self.last_spread = 0.0
        self.is_training = is_training
        self.max_frames = max_frames
        self.price_frames = deque(maxlen=max_frames)
        self.position_data_history = deque(maxlen=max_frames * 6)
        self.long_pnl_history = []
        self._best_ask = 0.0
        self._best_bid = 0.0
        self.action_space = Discrete(2) if action_space is None \
            else action_space
        self.ask_diff = 0.0
        self.bid_diff = 0.0
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
        self.max_episode_length_str = '10m'
        self.max_episode_length = timeparse(self.max_episode_length_str)
        self.max_summary = max_summary
        self.max_loss = max_loss
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
        self.current_trade = None
        self._position = None
        self.short_pnl = 0.0
        self.long_pnl = 0.0
        self.print_ascii_chart = print_ascii_chart
        self.position_repeat = 0
        self.trade_size = self.capital * (10/100)
        self.reset_count = 0
        self.levels = levels

        high = np.full(
            (sequence_length, depth * 2, 1),
            1.0,
            dtype=np.float32
        )
        low = np.full(
            (sequence_length, depth * 2, 1),
            0.0,
            dtype=np.float32
        )

        self.observation_space = Box(low, high, dtype=np.float32)

    @staticmethod
    def get_action_meanings():
        return ['NOOP']

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    @property
    def position(self):
        if self.current_trade:
            return self.current_trade.position_type
        else:
            return Positions.Flat

    @position.setter
    def position(self, action: np.float):
        self.last_position = self.position

        if self.should_change_position(action):
            self.change_position(action)

        if self.current_trade is None:
            raise Exception()

    @property
    def reward_ratio(self):
        return self._reward_ratio

    @reward_ratio.setter
    def reward_ratio(self, value):
        self._reward_ratio = value

    @property
    def done(self):
        return self._done

    @done.setter
    def done(self, value):
        traceback.print_stack()
        self._done = value

    def get_volatile_ranges(self):
        query = f'SELECT bbd FROM (SELECT STDDEV(best_bid) as bbd ' \
            f'from {self.channel_name} GROUP BY time({self.max_episode_length_str})) '\
            f'WHERE bbd > {self.min_std_dev};'

        ranges = self.query(query).get_points(self.channel_name)

        return DataFrame(ranges).set_index('time')

    def reset(self, **kwargs):
        self.reset_count += 1
        reset_count = self.reset_count

        if self.step_count > 0:
            alog.debug('##### reset ######')
            alog.info(alog.pformat(self.summary()))

        _kwargs = self._args['kwargs']
        del self._args['kwargs']
        _kwargs = {**self._args, **_kwargs, **kwargs}
        del _kwargs['self']
        new_instance = self.reset_class(**_kwargs)

        self.__dict__ = {**self.__dict__, **new_instance.__dict__}
        self.reset_count = reset_count

        self.observations = self._get_observation()

        try:
            self.get_observation()
        except StopIteration:
            self.observations = self._get_observation()
            self.get_observation()

        return self.last_observation

    def step_position(self, action):
        self.position = action
        self.current_trade.step(self.best_bid, self.best_ask)

        if self.current_trade.done:
            self.done = True

    def step(self, action):
        assert self.action_space.contains(action)

        self.step_position(action)

        self.reward += self.current_trade.reward

        self.step_count += 1

        if self.step_count >= self.max_episode_length:
            self.done = True

        observation = self.get_observation()

        reward = self.reset_reward()

        self.print_summary()

        return observation, reward, self.done, {}

    def print_summary(self):
        if settings.LOG_LEVEL == logging.DEBUG:
            if self.step_count % self.summary_interval == 0:
                alog.info(alog.pformat(self.summary()))

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
        self.bids = bids.reshape((2, bids.shape[1])).swapaxes(0, 1)

        asks = np.array([orderbook[1][0], orderbook[1][1]])
        self.asks = asks.reshape((2, asks.shape[1])).swapaxes(0, 1)

        self.position_data_history.appendleft(self.position_data)

        self.long_pnl_history.append(self.long_pnl)

        plot = self.plot_orderbook()

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
        bids = np.flip(self.bids[:10], 0)
        asks = self.asks

        for price, volume in bids:
            ax1.bar(price, volume, color='red', width=0.45)

        for price, volume in asks:
            ax1.bar(price, volume, color='blue', width=0.45)

        plt.ylim(0, self.top_limit)

        plt.xlim(bids[0, 0], asks[-1, 0])

        self.hide_ticks_and_values(ax1)

        fig.patch.set_visible(False)
        fig.canvas.draw()

        img = fig.canvas.renderer._renderer
        img = np.array(img)

        if self.print_ascii_chart:
            if self.step_count % self.summary_interval == 0:
                alog.info(AsciiImage(img, new_width=10))

        img = img[:, :, :3]
        return img

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

    @property
    def min_profit(self):
        if self.best_bid > 0:
            change = self.min_change / self.last_best_bid
            return (self.capital * change) + (-1 * self.capital * self.trading_fee)
        else:
            return 0.0

    def should_change_position(self, action):
        if self.current_trade is None:
            return True

        return self.position.value != action

    def long(self):
        if isinstance(self.current_trade, Trade):
            self.close_trade()
        if isinstance(self.current_trade, LongTrade):
            raise Exception('Already Long')

        if self.current_trade is None:
            self.current_trade = LongTrade(
                leverage=self.leverage,
                capital=self.capital,
                entry_price=self.best_bid,
                trading_fee=self.trading_fee,
                min_change=self.min_change,
                reward_ratio=self.reward_ratio,
                step_reward_ratio=self.step_reward_ratio,
                step_reward=self.step_reward,
                min_steps=self.min_steps
            )
            self.current_trade.step(self.best_bid, self.best_ask)

    def short(self):
        if isinstance(self.current_trade, Trade):
            self.close_trade()
        if isinstance(self.current_trade, ShortTrade):
            raise Exception('Already Long')

        if self.current_trade is None:
            self.current_trade = ShortTrade(
                leverage=self.leverage,
                capital=self.capital,
                entry_price=self.best_bid,
                trading_fee=self.trading_fee,
                min_change=self.min_change,
                step_reward_ratio=self.step_reward_ratio,
                reward_ratio=self.reward_ratio,
                step_reward=self.step_reward,
                min_steps=self.min_steps,
                ** self._args['kwargs']
            )
            self.current_trade.step(self.best_bid, self.best_ask)

    def flat(self):
        if isinstance(self.current_trade, Trade):
            self.close_trade()
        if isinstance(self.current_trade, FlatTrade):
            raise Exception('Already Flat')

        if self.current_trade is None:
            self.current_trade = FlatTrade(
                capital=self.trade_size,
                # entry_price=self.best_bid,
                entry_price=self.best_ask,
                leverage=self.leverage,
                min_change=self.min_change,
                min_steps=self.min_steps,
                reward_ratio=self.reward_ratio,
                step_reward=self.step_reward,
                step_reward_ratio=self.step_reward_ratio,
                trading_fee=self.trading_fee,
                **self._args['kwargs']
            )
            self.current_trade.step(self.best_bid, self.best_ask)

    def close_trade(self):
        trade: Trade = self.current_trade
        trade.close()

        self.reward += trade.reward

        if type(trade) != FlatTrade:
            self.capital += trade.capital - 1

        self.reward += self.current_trade.reward

        self.trades.append(trade)
        self.current_trade = None

    def change_position(self, action):
        action = int(action)

        if action == Positions.Long.value:
            self.long()
        elif action == Positions.Short.value:
            self.short()
        elif action == Positions.Flat.value:
            self.flat()

    def summary(self):
        summary_keys = [
            '_best_ask',
            '_best_bid',
            'capital',
            'leverage',
            'last_datetime',
            'step_count',
            'total_reward'
        ]

        summary = {key: self.__dict__[key] for key in
                   summary_keys}

        summary['position_history'] = \
            ''.join(self.position_history[-1 * self.max_summary:])

        summary['trades'] = [trade for trade in self.trades[-1 * self.max_summary:]
                             if type(trade) != FlatTrade]

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
        use_volatile_ranges=False,
        window_size='30s',
        is_training=False,
        print_ascii_chart=True,
        **kwargs
    )

    env.reset()

    trade_length = 120
    test_length = timeparse(test_span)

    while test_length > 0:
        for i in range(trade_length):
            env.step(random.randint(0, 2))
            test_length -= 1

    alog.info(alog.pformat(env.summary()))


if __name__ == '__main__':
    main()
