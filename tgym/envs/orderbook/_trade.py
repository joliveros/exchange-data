from yaml.representer import SafeRepresenter

from exchange_data import settings
from skimage import color
from tgym.envs.orderbook.ascii_image import AsciiImage
from tgym.envs.orderbook.utils import Positions
from typing import Callable
import alog
import logging
import matplotlib.pyplot as plt
import numpy as np
import yaml

class Logging(object):
    def __init__(self):
        yaml.add_representer(float, SafeRepresenter.represent_float)

    def yaml(self, value: dict):
        return yaml.dump(value)

class Trade(Logging):
    def __init__(
        self,
        entry_price: float,
        capital: float,
        trading_fee: float,
        position_type: Positions,
        min_change: float
    ):
        Logging.__init__(self)
        self.min_change = min_change
        self.max_increase_reward = 1.0
        self.max_steps_reward = 3
        self.positive_close_reward = 1.0
        self.steps_since_max = 0.0
        self.frame_width = 42
        self.capital = capital
        self.trading_fee = trading_fee
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

    @property
    def is_long(self):
        return self.position_type.value == Positions.Long.value

    @property
    def is_flat(self):
        return self.position_type.value == Positions.Flat.value

    @property
    def best_bid(self):
        return self.bids[0]

    @property
    def best_ask(self):
        return self.asks[0]

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
        if self.pnl > 0.0:
            self.reward += self.positive_close_reward
        else:
            self.reward += -1 * self.positive_close_reward

        if settings.LOG_LEVEL == logging.DEBUG:
            alog.info(f'{self.plot()}\n{self.yaml(self.summary())}')

    def step(self, best_bid: float, best_ask: float):
        self.clear_pnl()
        self.reward = 0.0
        self.position_length += 1
        self.bids = np.append(self.bids, [best_bid])
        self.asks = np.append(self.asks, [best_ask])
        pnl = self.pnl
        self.pnl_history = np.append(self.pnl_history, [pnl])

        if pnl > self.max_pnl or self.max_pnl == 0.0:
            self.clear_steps_since_max()
            self.max_pnl = pnl
            self.reward += self.max_increase_reward

        if self.position_length > 5 and pnl < self.max_pnl:
            self.reward += -1 * self.max_increase_reward
            alog.info(f'### long and less than max {self.reward}/{pnl}/{self.max_pnl} ###')

        if pnl > self.min_profit:
            self.reward += self.max_increase_reward
        else:
            self.reward += -1 * self.max_increase_reward / 10

        self.steps_since_max += 1

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
            f'{self.position_length}'

    def clear_pnl(self):
        self._pnl = None

    def clear_steps_since_max(self):
        self.steps_since_max = 0.0


class LongTrade(Trade):
    def __init__(self, **kwargs):
        Trade.__init__(self, position_type=Positions.Long, **kwargs)

    def close(self):
        self.capital += self.pnl
        super().close()

    @property
    def exit_price(self):
        return self.best_bid

    @property
    def pnl(self):
        if self._pnl:
            return self._pnl

        diff = self.exit_price - self.entry_price
        if self.entry_price == 0.0:
            change = 0.0
        else:
            change = diff / self.entry_price

        pnl = (self.capital * change) + \
                   (-1 * self.capital * self.trading_fee)

        return pnl


class ShortTrade(Trade):
    def __init__(self, **kwargs):
        Trade.__init__(self, position_type=Positions.Short, **kwargs)

    @property
    def exit_price(self):
        return self.best_ask

    @property
    def pnl(self):
        if self._pnl:
            return self._pnl

        diff = self.entry_price - self.exit_price

        if self.entry_price == 0.0:
            change = 0.0
        else:
            change = diff / self.entry_price

        pnl = (self.capital * change) + \
                   (-1 * self.capital * self.trading_fee)
        return pnl

    def close(self):
        self.capital += self.pnl
        super().close()
