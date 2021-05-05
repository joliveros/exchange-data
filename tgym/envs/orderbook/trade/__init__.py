import alog
import numpy as np
from matplotlib import pyplot as plt
from skimage import color

from exchange_data.trading import Positions
from tgym.envs.orderbook.utils import Logging
from tgym.envs.orderbook.ascii_image import AsciiImage


class Trade(Logging):
    def __init__(
        self,
        entry_price: float,
        capital: float,
        trading_fee: float,
        position_type: Positions,
        min_change: float,
        leverage: float = 1.0,
        reward_ratio: float = 1.0,
        step_reward_ratio: float = 1.0,
        step_reward: float = 1.0,
        min_steps: int = 10,
    ):
        Logging.__init__(self)
        self.trading_fee = trading_fee
        self.min_steps = min_steps
        self.step_reward_ratio = step_reward_ratio
        self.step_reward = step_reward
        self.reward_ratio = reward_ratio
        self.postive_pnl_reward = (1 - self.reward_ratio)
        self.leverage = leverage
        self.min_change = min_change
        self.max_increase_reward = 1.0
        self.max_steps_reward = 3
        self.positive_close_reward = 1.0
        self.steps_since_max = 0.0
        self.frame_width = 42
        self.capital = capital
        self.position_length = 0
        self.position_type = position_type
        self.entry_price = entry_price
        self.bids = np.array([])
        self.asks = np.array([])
        self.pnl_history = np.array([])
        self._pnl = None
        self.max_pnl = 0.0
        self._reward = 0.0
        self.total_reward = 0.0
        self.done = False

    @property
    def is_long(self):
        return self.position_type.value == Positions.Long.value

    @property
    def is_flat(self):
        return self.position_type.value == Positions.Flat.value

    @property
    def best_bid(self):
        return self.bids[-1]

    @property
    def best_ask(self):
        if self.asks.shape[0] > 0:
            return self.asks[-1]
        return 0.0

    @property
    def pnl(self):
        raise NotImplementedError

    @property
    def reward(self):
        reward = self._reward
        return reward

    @reward.setter
    def reward(self, value):
        self._reward = value

    @property
    def min_profit(self):

        change = self.min_change / self.exit_price

        pnl = (self.capital * change) + \
                   (-1 * self.capital * self.trading_fee)

        return pnl

    def close(self):
        #if self.pnl >= self.min_profit:
            #self.reward += self.reward_ratio
        #else:
            #self.reward += self.reward_ratio * -1.0

        #self.total_reward += self.reward
        pass

    def step(self, best_bid: float, best_ask: float):
        self.clear_pnl()
        self.reward = 0.0
        self.position_length += 1
        self.bids = np.append(self.bids, [best_bid])
        self.asks = np.append(self.asks, [best_ask])

        pnl = self.pnl
        last_pnl = 0.0

        if len(self.pnl_history) > 0:
            last_pnl = self.pnl_history[-1]

        self.pnl_history = np.append(self.pnl_history, [pnl])

        pnl_delta = self.pnl - last_pnl

        # if pnl_delta > 0.0:
        #     self.reward += self.step_reward * self.step_reward_ratio

        if self.pnl > self.min_profit:
            self.reward += self.step_reward * self.step_reward_ratio
        #else:
            #self.reward += self.step_reward * self.step_reward_ratio * -1.0

        #if pnl_delta < 0.0 and self.position_length > 6:
           # self.reward -= self.step_reward * self.step_reward_ratio
           # self.done = True

        self.total_reward += self.reward

    def plot(self):
        fig, price_frame = plt.subplots(1, 1, figsize=(2, 1), dpi=self.frame_width)
        avg = np.add(self.bids, self.asks) / 2
        price_frame.plot(avg, color='black')
        pnl_frame = price_frame.twinx()
        pnl_frame.plot(self.pnl_history, color='green')
        plt.autoscale(tight=True)
        self.hide_ticks_and_values(price_frame)
        self.hide_ticks_and_values(pnl_frame)
        fig.patch.set_visible(False)
        fig.canvas.draw()

        img = fig.canvas.renderer._renderer
        img = np.array(img)
        img = color.rgb2gray(img)

        plt.close(fig)

        return str(AsciiImage(img, new_width=21)) + '\n'

    def summary(self):
        summary_keys = [
            'position_length',
            'entry_price',
        ]

        summary = {key: self.__dict__[key] for key in
                   summary_keys}
        summary['reward'] = self.reward
        summary['exit_price'] = float(self.exit_price)
        summary['pnl'] = float(self.pnl)
        summary['entry_price'] = float(self.entry_price)
        summary['position_type'] = str(self.position_type)
        summary['capital'] = float(self.capital)

        return summary

    def hide_ticks_and_values(self, frame):
        frame.axis('off')

    def __repr__(self):
        return f'{self.position_type}/{self.pnl}/{self.entry_price}/{self.exit_price}/' \
            f'{self.total_reward}/{self.position_length}'

    def clear_pnl(self):
        self._pnl = None

    def clear_steps_since_max(self):
        self.steps_since_max = 0.0
