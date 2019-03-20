from abc import ABC
from gym.spaces import Discrete

from exchange_data.utils import NoValue
from tgym.envs import OrderBookTradingEnv


class Positions(NoValue):
    Flat = 0
    Long = 1


class LongOrderBookTradingEnv(OrderBookTradingEnv, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_space = Discrete(2)

    def change_position(self, action):
        self.last_position = self.position

        if action == Positions.Long.value:
            self.long()
        elif action == Positions.Flat.value:
            self.flat()
