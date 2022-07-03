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

    @property
    def pnl(self):
        diff = self.entry_price - self.exit_price

        if self.entry_price == 0.0:
            change = 0.0
        else:
            change = diff / self.entry_price

        pnl = (self.capital * change) * self.leverage
        fee = (self.fee * self.capital) + (self.fee * (self.capital * (1 + change)))
        pnl = pnl + fee
        return pnl

    def close(self):
        super().close()

        self.reward_for_pnl()

        if self.total_reward == 0.0:
            self.reward += -0.001
            self.total_reward += self.reward

        self.capital += self.pnl

    def reward_for_pnl(self):
        if self.position_length > self.max_short_position_length:
            pnl = self.pnl / self.position_length
        else:
            pnl = self.pnl


        if pnl > self.min_change:
            self.reward += pnl
        else:
            if pnl > 0.0:
                self.reward += pnl * -1
            else:
                self.reward += pnl

        self.total_reward += self.reward

        self.last_pnl = pnl
