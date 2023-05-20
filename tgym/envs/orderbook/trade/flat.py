from exchange_data.trading import Positions
from tgym.envs.orderbook.trade import Trade
import numpy as np
import alog


class FlatTrade(Trade):
    def __init__(
        self,
        min_flat_change=0.0,
        max_flat_position_length=0,
        fee_ratio=1.0,
        short_reward_enabled=False,
        **kwargs
    ):
        super().__init__(
            position_type=Positions.Flat,
            **kwargs
        )

        if self.is_test:
            max_flat_position_length = 0

        self.fee_ratio = fee_ratio
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

        return pnl + ( fee * self.fee_ratio )

    def step(self, best_bid: float, best_ask: float):
        self.position_length += 1
        self.bids = np.append(self.bids, [best_bid])
        self.asks = np.append(self.asks, [best_ask])

        if self.position_length >= self.max_flat_position_length > 0:
            self.done = True

        self.append_pnl_history()

    def reward_for_pnl(self):
        pnl = self.pnl / self.position_length

        if self.position_length >= self.max_flat_position_length > 0:
            reward = pnl / self.position_length
        else:
            reward = 0

        self.reward += reward * self.reward_ratio

        self.last_pnl = pnl

    def close(self):
        self.reward_for_pnl()

        if self.position_length >= self.max_flat_position_length > 0:
            self.done = True

        super().close()
