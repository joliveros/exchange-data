from exchange_data.trading import Positions
from tgym.envs.orderbook.trade import Trade
import numpy as np
import alog


class NoRewardFlatTrade(Trade):
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
