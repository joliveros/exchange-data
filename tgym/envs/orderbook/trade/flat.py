from exchange_data.trading import Positions
from tgym.envs.orderbook.trade import Trade
import numpy as np
import alog


class FlatTrade(Trade):
    def __init__(self, short_reward_enabled=False, **kwargs):
        super().__init__(
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

        fee = 0.0
        capital = 1.0

        return (capital * (change * self.leverage)) + fee

    def step(self, best_bid: float, best_ask: float):
        self.reward = 0.0
        self.position_length += 1
        self.bids = np.append(self.bids, [best_bid])
        self.asks = np.append(self.asks, [best_ask])

        self.append_pnl_history()

    def close(self):
        if self.short_reward_enabled:
            self.reward_for_pnl()
