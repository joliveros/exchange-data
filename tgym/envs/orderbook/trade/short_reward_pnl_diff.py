from tgym.envs.orderbook.trade.short import ShortTrade
import alog


class ShortRewardPnlDiffTrade(ShortTrade):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, *args, **kwargs):
        super().step(*args, **kwargs)

        self.reward_for_pnl()

    def reward_for_pnl(self):
        if self.position_length < self.max_short_position_length \
                or self.max_short_position_length == 0:
            pnl = self.pnl
            diff = pnl - self.last_pnl

            if diff > 0:
                self.reward = diff * self.reward_ratio

            self.total_reward += self.reward

            self.last_pnl = pnl
