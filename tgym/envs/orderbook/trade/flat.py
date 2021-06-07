import alog
from exchange_data.trading import Positions
from tgym.envs.orderbook.trade import Trade

import numpy as np


class FlatTrade(Trade):
    def __init__(self, **kwargs):
        Trade.__init__(self, position_type=Positions.Flat, **kwargs)

    @property
    def exit_price(self):
        return (self.best_ask + self.best_bid) / 2

    @property
    def pnl(self):
        if self.entry_price == 0.0:
            change = 0.0
        else:
            change = self.raw_pnl / self.entry_price

        return self.capital * (change * self.leverage) * -1

    def close(self):
        self.reward += self.pnl * -1 * self.reward_ratio
