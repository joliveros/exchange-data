from collections import deque
from typing import List

from exchange_data.streamers._bitmex import BitmexStreamer
from gym.spaces import Box
from tgym.core import Env
from tgym.envs.orderbook import Actions, Positions

import alog
import numpy as np

POSITIONS = Positions()
ACTIONS = Actions()


class OrderBookTradingEnv(Env, BitmexStreamer):
    """
    Orderbook based trading environment.
    """

    def __init__(
        self,
        episode_length=1000,
        trading_fee=0,
        time_fee=0,
        max_frames=10,
        **kwargs
    ):
        Env.__init__(self)
        BitmexStreamer.__init__(self, **kwargs)
        self.max_frames = max_frames
        action_space = Box(0, 1.0, shape=(3, ), dtype=np.float32)
        alog.info(action_space)

        self._trading_fee = trading_fee
        self._time_fee = time_fee
        self._episode_length = episode_length
        self.n_actions = len(ACTIONS)
        self.frames = deque(maxlen=self.max_frames)
        # index, orderbook = self.compose_window()

        self.reset()

    def reset(self):
        self._iteration = 0
        self._total_reward = 0
        self._total_pnl = 0
        self._position = Positions.Flat
        self._entry_price = 0
        self._exit_price = 0
        self._closed_plot = False

        observation = self._get_observation()
        self.state_shape = observation.shape
        self._action = self._actions['hold']
        return observation

    def step(self, action):
        assert any([(action == x).all() for x in ACTIONS])

        self._action = action
        self._iteration += 1
        done = False
        instant_pnl = 0
        info = {}
        reward = -self._time_fee

        if all(action == ACTIONS.Buy):
            reward -= self._trading_fee
            if all(self._position == POSITIONS.Flat):
                self._position = POSITIONS.Long
                self._entry_price = calc_spread(
                    self._prices_history[-1], self._spread_coefficients)[1]  # Ask

            elif all(self._position == self._positions['short']):
                self._exit_price = calc_spread(
                    self._prices_history[-1], self._spread_coefficients)[1]  # Ask
                instant_pnl = self._entry_price - self._exit_price
                self._position = self._positions['flat']
                self._entry_price = 0

        elif all(action == self._actions['sell']):
            reward -= self._trading_fee
            if all(self._position == self._positions['flat']):
                self._position = self._positions['short']
                self._entry_price = calc_spread(
                    self._prices_history[-1], self._spread_coefficients)[0]  # Bid
            elif all(self._position == self._positions['long']):
                self._exit_price = calc_spread(
                    self._prices_history[-1], self._spread_coefficients)[0]  # Bid
                instant_pnl = self._exit_price - self._entry_price
                self._position = self._positions['flat']
                self._entry_price = 0

        reward += instant_pnl
        self._total_pnl += instant_pnl
        self._total_reward += reward

        # Game over logic
        try:
            self._prices_history.append(self._data_generator.next())
        except StopIteration:
            done = True
            info['status'] = 'No more data.'
        if self._iteration >= self._episode_length:
            done = True
            info['status'] = 'Time out.'
        if self._closed_plot:
            info['status'] = 'Closed plot'

        observation = self._get_observation()
        return observation, reward, done, info

    def _get_observation(self):
        index, orderbook = next(self)
        index = np.array([index])

        if isinstance(orderbook, list):
            orderbook = np.array(orderbook).flatten()

        alog.info(index)
        alog.info(orderbook)
        frame = np.concatenate((index, orderbook))

        self.frames.append(frame)

        return np.concatenate(
            self.frames,
            np.array([
                np.array([self._entry_price]),
                np.array(self._position)
            ])
        )

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
