from exchange_data.trading import Positions
from tgym.envs.orderbook.trade import Trade


class ShortTrade(Trade):
    def __init__(self, **kwargs):
        Trade.__init__(self, position_type=Positions.Short, **kwargs)

    @property
    def exit_price(self):
        return self.best_ask

    @property
    def pnl(self):
        diff = self.entry_price - self.exit_price

        if self.entry_price == 0.0:
            change = 0.0
        else:
            change = diff / self.entry_price

        fee = (-1 * self.capital * self.trading_fee)

        pnl = (self.capital * (change * self.leverage)) + fee

        return pnl

    def close(self):
        self.capital += self.pnl
        super().close()
