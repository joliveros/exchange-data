from exchange_data.trading import Positions
from tgym.envs.orderbook.trade import Trade

import alog


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
        diff = self.exit_price - self.entry_price

        if self.entry_price == 0.0:
            change = 0.0
        else:
            change = diff / self.entry_price
        pnl = (self.capital * (change * self.leverage)) + \
                   (-1 * self.capital * self.trading_fee)

        return pnl