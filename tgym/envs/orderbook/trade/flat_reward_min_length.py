from tgym.envs.orderbook.trade import FlatTrade
import alog


class FlatRewardMinLength(FlatTrade):
    def __init__(self, min_flat_position_length, **kwargs):
        super().__init__(**kwargs)
        self.min_flat_position_length = min_flat_position_length

    def step(self, *args, **kwargs):
        super().step(*args, **kwargs)

        self.reward_for_pnl()

    def reward_for_pnl(self):
        pnl = self.pnl

        if self.position_length > self.min_flat_position_length \
                or self.min_flat_position_length == 0:

            diff = pnl - self.last_pnl

            if diff > 0 and pnl > 0:
                self.reward = diff

            self.total_reward += self.reward

            self.last_pnl = pnl
