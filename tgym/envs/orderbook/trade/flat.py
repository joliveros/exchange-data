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
        return 0.0
        diff = self.entry_price - self.exit_price

        if self.entry_price == 0.0:
            change = 0.0
        else:
            change = diff / self.entry_price

        pnl = (self.capital * change) + \
                   (-1 * self.capital * self.trading_fee)
        return pnl

    def step(self, best_bid: float, best_ask: float):
        self.reward = 0.0
        self.position_length += 1
        self.bids = np.append(self.bids, [best_bid])
        self.asks = np.append(self.asks, [best_ask])

        # if self.position_length > 30:
        #     # raise TrialPruned()
        #     self.done = True

        # if len(self.asks) > 2:
        #     if self.best_ask != self.asks[-2]:
        #         self.reward -= self.reward_ratio
        #         # self.done = True
            # else:
        self.reward -= self.flat_reward

        self.total_reward += self.reward

    def close(self):
        return
        if settings.LOG_LEVEL == logging.DEBUG:
            alog.info(f'{self.yaml(self.summary())}')


