import logging

import alog
import numpy as np

from exchange_data import settings
from exchange_data.trading import Positions
from tgym.envs.orderbook.trade import Trade


class FlatTrade(Trade):
    def __init__(self, **kwargs):
        Trade.__init__(self, position_type=Positions.Flat, **kwargs)

    @property
    def exit_price(self):
        return (self.best_ask + self.best_bid) / 2

    @property
    def pnl(self):
        diff = self.exit_price - self.entry_price

        if self.entry_price == 0.0:
            change = 0.0
        else:
            change = diff / self.entry_price
        pnl = (self.capital * (change * self.leverage)) + \
                   (-1 * self.capital * self.trading_fee)

        return pnl

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

        if pnl_delta > 0.0:
            self.reward -= self.step_reward * self.step_reward_ratio
        else:
            self.reward += self.step_reward * self.step_reward_ratio

        self.total_reward += self.reward

    def close(self):
        return
        if settings.LOG_LEVEL == logging.DEBUG:
            alog.info(f'{self.yaml(self.summary())}')


