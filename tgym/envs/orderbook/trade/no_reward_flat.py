from tgym.envs.orderbook.trade.flat import FlatTrade
import alog


class NoRewardFlatTrade(FlatTrade):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

    def step(self, *args, **kwargs):
        super().step(*args, **kwargs)

    def reward_for_pnl(self):
        pass

    def close(self):
        super().close()
