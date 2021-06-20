from exchange_data.trading import Positions
from tgym.envs.orderbook.trade import Trade


class FlatTrade(Trade):
    def __init__(self, **kwargs):
        Trade.__init__(self, position_type=Positions.Flat, **kwargs)

    @property
    def exit_price(self):
        return self.best_bid

    @property
    def pnl(self):
        if self.entry_price == 0.0:
            change = 0.0
        else:
            change = self.raw_pnl / self.entry_price

        fee = (-1 * self.capital * self.trading_fee)

        return (self.capital * (change * self.leverage)) + fee

    def close(self):
        self.reward += self.pnl * self.reward_ratio
