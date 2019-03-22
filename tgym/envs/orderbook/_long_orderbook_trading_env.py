from abc import ABC
from exchange_data.utils import NoValue
from gym.spaces import Discrete
from tgym.envs.orderbook._orderbook import OrderBookTradingEnv


class Positions(NoValue):
    Flat = 0
    Long = 1


class LongOrderBookTradingEnv(OrderBookTradingEnv, ABC):
    def __init__(self, **kwargs):
        action_space = Discrete(2)
        OrderBookTradingEnv.__init__(self, action_space=action_space, **kwargs)

    def change_position(self, action):
        self.last_position = self.position

        if action == Positions.Long.value:
            self.long()
        elif action == Positions.Flat.value:
            self.flat()

    # def should_change_position(self, action):
    #     if action == Positions.Flat.value:
    #         self.reward += 0.5
    #
    #     return self.position.value != action
