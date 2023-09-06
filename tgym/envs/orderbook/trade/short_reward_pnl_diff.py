from tgym.envs.orderbook.trade.short import ShortTrade
import alog


class ShortRewardPnlDiffTrade(ShortTrade):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def step(self, *args, **kwargs):
        super().step(*args, **kwargs)

        self.reward_for_pnl()

    def reward_for_pnl(self):
        # if self.position_length < self.max_short_position_length \
        #         or self.max_short_position_length == 0:
        pnl = self.pnl

        if pnl <= self.min_change:
            self.done = True

        if self.last_pnl != 0.0:
            diff = pnl - self.last_pnl

            if diff > 0 and pnl > 0:
                self.reward = diff

            self.total_reward += self.reward

        self.last_pnl = pnl
