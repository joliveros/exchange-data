from exchange_data.trading import Positions
from tgym.envs.orderbook.trade import Trade
import numpy as np
import alog


class FlatTrade(Trade):
    def __init__(
        self,
        min_flat_change=0.0,
        max_flat_position_length=2,
        short_reward_enabled=False,
        **kwargs
    ):
        super().__init__(
            position_type=Positions.Flat,
            **kwargs
        )
        self.min_flat_change = min_flat_change
        self.max_flat_position_length = max_flat_position_length
        self.short_reward_enabled = short_reward_enabled

    @property
    def exit_price(self):
        return self.best_bid

    @property
    def pnl(self):
        if self.entry_price == 0.0:
            change = 0.0
        else:
            change = self.raw_pnl / self.entry_price

        pnl = (self.capital * change) * self.leverage

        fee = (self.fee * self.capital) + (self.fee * (self.capital * (1 + change)))

        return pnl + fee

    def step(self, best_bid: float, best_ask: float):
        self.position_length += 1
        self.bids = np.append(self.bids, [best_bid])
        self.asks = np.append(self.asks, [best_ask])

        ## self.reward_for_pnl()

        self.append_pnl_history()

        if self.position_length >= self.max_flat_position_length > 0:
            self.done = True

    def close(self):
        self.reward_for_pnl()
        
        if self.total_reward == 0.0:
            # self.reward = -0.001
            self.total_reward = self.reward
