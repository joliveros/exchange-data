import alog

from exchange_data.trading import Positions
from tgym.envs.orderbook.trade import Trade


class ShortTrade(Trade):
    fee = 0.0

    def __init__(self, **kwargs):
        super().__init__(position_type=Positions.Short, **kwargs)

    @property
    def fee(self):
        fee = 0.0

        if len(self.pnl_history) == 1 or self.closed:
            fee = -1 * self.capital * self.leverage * self.trading_fee

        return fee

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

        pnl = ((self.capital * change) * self.leverage) + (self.fee * 2)

        return pnl

    def close(self):
        super().close()

        pnl = self.pnl
        if pnl < 0.0:
            _pnl = abs(pnl)
            self.reward = (_pnl ** (1/4)) * -1
        else:
            self.reward = pnl ** (1/4)

        self.capital += pnl

