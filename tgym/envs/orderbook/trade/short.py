import alog

from exchange_data.trading import Positions
from tgym.envs.orderbook.trade import Trade


class ShortTrade(Trade):
    def __init__(self, max_short_position_length=-1, **kwargs):
        super().__init__(position_type=Positions.Short, **kwargs)
        self.max_short_position_length = max_short_position_length

    @property
    def exit_price(self):
        return self.best_ask

    def step(self, *args):
        super().step(*args)

        if self.position_length >= self.max_short_position_length and \
            self.max_short_position_length > 0:
            alog.info(args)
            alog.info(f'### environment should be done {self.position_length} ###')
            self.done = True

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
