from exchange_data.trading import Positions
from tgym.envs.orderbook.trade import Trade
import numpy as np
import alog


class FlatTrade(Trade):
    def __init__(self, short_reward_enabled=False, **kwargs):
        Trade.__init__(
            self,
            position_type=Positions.Flat,
            **kwargs
        )
        self.short_reward_enabled = short_reward_enabled

    @property
    def exit_price(self):
        return self.best_ask

    @property
    def pnl(self):
        if self.entry_price == 0.0:
            change = 0.0
        else:
            change = self.raw_pnl / self.entry_price

        # fee = (-1 * self.capital * self.leverage * self.trading_fee)
        fee = 0.0
        capital = 1.0

        return ((capital * (change * self.leverage)) + fee)

    def step(self, best_bid: float, best_ask: float):
        self.reward = 0.0
        self.position_length += 1
        self.bids = np.append(self.bids, [best_bid])
        self.asks = np.append(self.asks, [best_ask])

        pnl = self.pnl
        last_pnl = 0.0

        if len(self.pnl_history) > 0:
            last_pnl = self.pnl_history[-1]

        self.pnl_history = np.append(self.pnl_history, [pnl])

    def close(self):
        if self.short_reward_enabled:
            self.reward += self.pnl * self.reward_ratio
