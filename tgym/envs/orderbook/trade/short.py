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
        if self.position_length >= self.max_short_position_length:
            self.done = True

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
        # if self.position_length <= self.max_short_position_length:
        #     pnl = self.pnl
        # else:
        #     pnl = self.pnl / self.position_length

        pnl = self.pnl

        if pnl > self.min_change:
            reward = pnl / self.position_length
        else:
            reward = abs(pnl) * -1

        self.total_reward += reward * self.reward_ratio

        self.last_pnl = pnl

        if self.position_length >= self.max_short_position_length:
            self.done = True
