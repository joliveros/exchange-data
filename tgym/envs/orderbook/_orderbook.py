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
        trading_fee=.1,
        time_fee=0.1,
        max_frames=12,
        random_start_date=True,
        orderbook_depth=21,
        window_size='5m30s',
        sample_interval='5s',
        **kwargs
    ):
        kwargs['random_start_date'] = random_start_date
        kwargs['orderbook_depth'] = orderbook_depth
        kwargs['window_size'] = window_size
        kwargs['sample_interval'] = sample_interval
        self._args = locals()
        del self._args['self']

        kwargs['channel_name'] = \
            f'XBTUSD_OrderBookFrame_depth_{orderbook_depth}'

        Env.__init__(self)
        BitmexStreamer.__init__(self, **kwargs)

        self.last_timestamp = 0
        self.last_datetime = None
        self._position_pnl = 0
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
        self.time_fee = time_fee
        self.total_pnl = 0
        self.total_reward = 0
        self.trading_fee = trading_fee
        self.max_position_pnl = 0.0
        self.max_negative_pnl_factor = -0.01
        self.max_position_duration = 48
        self.max_pnl = 0.0
        self.position_history = []
        self.bid_diff = 0.0
        self.ask_diff = 0.0
        self.last_observation = None
        self.last_best_ask = None
        self.last_best_bid = None
        self.index_diff = None
        self._best_bid = None
        self._best_ask = None
        self.trades = []
        self.last_price_diff = 0.0

        high = np.full(
            (self.max_frames * (7 + 4 * self.orderbook_depth), ),
            np.inf
        )

        self.observation_space = Box(-high, high, dtype=np.float32)

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
        if self.step_count > 0:
            alog.info(alog.pformat(self.summary()))

        alog.info('### env reset ###')

        kwargs = self._args['kwargs']
        del self._args['kwargs']
        kwargs = {**self._args, **kwargs}
        new_instance = OrderBookTradingEnv(**kwargs)
        self.__dict__ = new_instance.__dict__

        try:
            for i in range(self.max_frames):
                self.get_observation()
        except (OutOfFramesException, TypeError):
            return self.reset()

        return self.last_observation

    def step(self, action):
        self.reset_reward()
        assert self.action_space.contains(action)

        done = False

        self.reward = -self.time_fee

        if self.should_change_position(action):
            self.change_position(action)
        else:
            if self.position_pnl > 0:
                self.reward += self.position_pnl / self.step_count

        position_pnl = self.position_pnl

        if position_pnl > self.max_position_pnl:
            self.max_position_pnl = position_pnl

        if position_pnl > 0:
            self.reward += position_pnl * 4.0

        if self.total_pnl > self.max_pnl:
            self.max_pnl = self.total_pnl

        if self.step_count >= self.max_position_duration \
            and self.total_pnl < self.step_count / 2:
            if self.position == Positions.Flat:
                done = True

        if self.step_count >= self.episode_length:
            done = True

        if self.out_of_frames_counter > 30 and not done:
            done = True

        if not done:
            self.step_count += 1
        try:
            observation = self.get_observation()
        except (OutOfFramesException, TypeError):
            observation = self.last_observation
            done = True

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
    def position_data(self):
        data_keys = [
            '_position_pnl',
            'ask_diff',
            'bid_diff',
            'index_diff',
            'max_position_pnl',
            'total_pnl'
        ]

        data = {key: self.__dict__[key] for key in
                   data_keys}

        data['position'] = self.position.value

        return np.array(list(data.values()))

    def local_fromtimestamp(self, value):
        return datetime.utcfromtimestamp(value/10**9)\
            .astimezone(tz.tzlocal())

    def get_observation(self):
        if self.last_observation is not None:
            self.last_best_ask = self.best_ask
            self.last_best_bid = self.best_bid
            self.last_price_diff = (self.last_best_ask + self.last_best_bid)/2 - \
                (self.best_ask + self.best_bid)/2
            self.reward -= abs(self.last_price_diff)

        time, index, orderbook = next(self)
        self.position_history.append(self.position.name[0])
        self.last_timestamp = time
        self.last_datetime = str(self.local_fromtimestamp(time))

        self.last_index = index
        self.last_orderbook = orderbook = np.array(orderbook)

        if self.last_observation is not None:
            self.bid_diff = self.best_bid - self.last_best_bid
            self.ask_diff = self.best_ask - self.last_best_ask

        idx_bbid_diff = self.best_ask + self.best_bid

        self.index_diff = ((index * 2) - idx_bbid_diff) / 2

        position_data = self.position_data

        frame = np.concatenate((position_data, orderbook.flatten()))

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

        pnl = self.short_pnl
        self.trades.append(Trade(
            self.position.name[0],
            pnl, self.entry_price,
            self.best_bid
        ))
        self.total_pnl += pnl
        self.reward += self.total_pnl * 4
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
        self.reward += self.total_pnl * 4
        self.entry_price = 0.0

    @property
    def short_pnl(self):
        if self.entry_price == 0.0 or self.best_bid == 0:
            return 0.0 - self.trading_fee
        if self.entry_price > 0.0:
            pnl = self.entry_price - self.best_bid - self.trading_fee
        else:
            pnl = 0.0
        return pnl

    @property
    def long_pnl(self):
        if self.entry_price == 0.0 or self.best_ask == 0.0:
            return 0.0 - self.trading_fee
        if self.entry_price > 0.0:
            pnl = self.best_ask - self.entry_price - self.trading_fee
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
        self.charge_trading_fee()
        if self.is_short:
            self.close_short()

        self.position = Positions.Long
        # alog.info(f'set long entry price {self.best_ask}')
        self.entry_price = self.best_ask

    def short(self):
        if self.is_short:
            raise Exception('Already short.')
        self.charge_trading_fee()
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
            '_position_pnl',
            'last_datetime',
            'max_pnl',
            'max_position_pnl',
            'step_count',
            'total_pnl',
            'total_reward',
        ]

        summary = {key: self.__dict__[key] for key in
                   summary_keys}

        summary['position_history'] = ''.join(self.position_history[-45:])
        summary['trades'] = self.trades[-45:]

        if self.step_count % self.max_position_duration == 0:
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

    env.reset()

    for i in range(timeparse(test_span)):
        env.step(Positions.Long.value)
        alog.info(alog.pformat(env.summary()))
        sleep(0.33)

    env.step(Positions.Flat.value)


if __name__ == '__main__':
    main()
