from abc import ABC
from tgym.envs.orderbook._orderbook import OrderBookTradingEnv


class OrderBookTradingEnvRLLib(OrderBookTradingEnv, ABC):
    def __init__(self, config: dict):
        OrderBookTradingEnv.__init__(self, **config)



