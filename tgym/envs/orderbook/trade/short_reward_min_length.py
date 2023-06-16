from tgym.envs.orderbook.trade.short import ShortTrade
import alog


class ShortRewardMinLength(ShortTrade):
    def __init__(self, min_position_length, **kwargs):
        super().__init__(**kwargs)
        self.min_position_length = min_position_length

    def step(self, *args, **kwargs):
        super().step(*args, **kwargs)

        self.reward_for_pnl()

    def reward_for_pnl(self):
        if self.position_length > self.min_position_length \
                or self.min_position_length == 0:
            pnl = self.pnl
            diff = pnl - self.last_pnl

            if diff > 0 and pnl > 0:
                self.reward = diff

            self.total_reward += self.reward

            self.last_pnl = pnl
