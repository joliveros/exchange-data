import alog

from exchange_data.trading import Positions
from tgym.envs.orderbook.trade import Trade


class ShortTrade(Trade):
    def __init__(self, max_short_position_length=0, **kwargs):
        super().__init__(position_type=Positions.Short, **kwargs)
        self.max_short_position_length = max_short_position_length

    @property
    def exit_price(self):
        return self.best_ask

    def step(self, *args):
        # if self.position_length >= self.max_short_position_length > 0:
        #    self.done = True

        self.reward_for_pnl()

        super().step(*args)

    @property
    def pnl(self):
        diff = self.entry_price - self.exit_price

        if self.entry_price == 0.0 or self.exit_price == 0.0:
            change = 0.0
        else:
            change = diff / self.entry_price

        pnl = (self.capital * change) * self.leverage
        fee = (self.fee * self.capital) + (self.fee * (self.capital * (1 + change)))
        pnl = pnl + fee

        return pnl

    def close(self):
        self.reward_for_pnl()
        self.capital += self.pnl
        super().close()

    def reward_for_pnl(self):
        pnl = self.pnl

        if pnl >= self.min_change:
            self.reward = pnl

        self.total_reward += self.reward

        self.last_pnl = pnl
