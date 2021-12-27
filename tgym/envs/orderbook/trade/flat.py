from exchange_data.trading import Positions
from tgym.envs.orderbook.trade import Trade
import numpy as np
import alog


class FlatTrade(Trade):
    def __init__(self, max_flat_position_length=2, short_reward_enabled=False,
                 **kwargs):
        super().__init__(
            position_type=Positions.Flat,
            **kwargs
        )
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
        fee = self.fee * 2

        return pnl + fee

    def step(self, best_bid: float, best_ask: float):
        self.position_length += 1
        self.bids = np.append(self.bids, [best_bid])
        self.asks = np.append(self.asks, [best_ask])

        self.append_pnl_history()

        # self.reward_for_pnl()

        # if self.position_length > self.max_flat_position_length:
        #     self.reward_for_pnl()

    def reward_for_pnl(self):
        pnl = self.pnl

        if pnl < 0:
            _pnl = abs(pnl)
            self.reward = (_pnl ** (1 / 4)) * -1
        else:
            self.reward = (pnl ** (1 / 4))

    def close(self):
        if self.short_reward_enabled:
            self.reward_for_pnl()
