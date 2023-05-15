from tgym.envs.orderbook.trade.flat import FlatTrade


class FlatRewardPnlDiffTrade(FlatTrade):

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(
            **kwargs
        )

    def step(self, *args, **kwargs):
        super().step(*args, **kwargs)

    def reward_for_pnl(self):
        pnl = self.pnl
        diff = pnl - self.last_pnl

        self.reward += diff * self.reward_ratio

        self.last_pnl = pnl

