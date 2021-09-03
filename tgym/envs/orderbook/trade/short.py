import alog

from exchange_data.trading import Positions
from tgym.envs.orderbook.trade import Trade


class ShortTrade(Trade):
    def __init__(self, **kwargs):
        super().__init__(position_type=Positions.Short, **kwargs)

    @property
    def exit_price(self):
        return self.best_bid

    @property
    def pnl(self):
        diff = self.entry_price - self.exit_price

        if self.entry_price == 0.0:
            change = 0.0
        else:
            change = diff / self.entry_price

        pnl = (self.capital * change) * self.leverage
        fee = self.fee * 2
        pnl = pnl + fee
        return pnl

    def close(self):
        super().close()

        self.reward_for_pnl()

        self.capital += self.pnl
