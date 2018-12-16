from tgym.core import Env
import logging
import numpy as np


class OrderBookTradingEnv(Env):
    """
    Orderbook based trading environment.
    """

    _actions = {
        'hold': np.array([1, 0, 0]),
        'buy': np.array([0, 1, 0]),
        'sell': np.array([0, 0, 1])
    }

    _positions = {
        'flat': np.array([1, 0, 0]),
        'long': np.array([0, 1, 0]),
        'short': np.array([0, 0, 1])
    }

    def __init__(self, episode_length=1000,
                 trading_fee=0, time_fee=0, history_length=2):
        """Initialisation function

        Args:
            episode_length (int): number of steps to play the game for
            trading_fee (float): penalty for trading
            time_fee (float): time fee
            history_length (int): number of historical states to stack in the
                observation vector.
        """

        super().__init__()
        assert history_length > 0
        self._first_render = True
        self._trading_fee = trading_fee
        self._time_fee = time_fee
        self._episode_length = episode_length
        self.n_actions = 3
        self._prices_history = []
        self._history_length = history_length
        self.reset()

    def reset(self):
        """Reset the trading environment. Reset rewards

        Returns:
            observation (numpy.array): observation of the state
        """
        self._iteration = 0
        self._total_reward = 0
        self._total_pnl = 0
        self._position = self._positions['flat']
        self._entry_price = 0
        self._exit_price = 0
        self._closed_plot = False

        # for i in range(self._history_length):
        #     self._prices_history.append(self._data_generator.next())

        observation = self._get_observation()
        self.state_shape = observation.shape
        self._action = self._actions['hold']
        return observation

    def step(self, action):
        """Take an action (buy/sell/hold) and computes the immediate reward.

        Args:
            action (numpy.array): Action to be taken, one-hot encoded.

        Returns:
            tuple:
                - observation (numpy.array): Agent's observation of the current environment.
                - reward (float) : Amount of reward returned after previous action.
                - done (bool): Whether the episode has ended, in which case further step() calls will return undefined results.
                - info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).

        """

        assert any([(action == x).all() for x in self._actions.values()])
        self._action = action
        self._iteration += 1
        done = False
        instant_pnl = 0
        info = {}
        reward = -self._time_fee
        if all(action == self._actions['buy']):
            reward -= self._trading_fee
            if all(self._position == self._positions['flat']):
                self._position = self._positions['long']
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
        """Concatenate all necessary elements to create the observation.

        Returns:
            numpy.array: observation array.
        """
        return np.concatenate(
            [prices for prices in self._prices_history[-self._history_length:]] +
            [
                np.array([self._entry_price]),
                np.array(self._position)
            ]
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
