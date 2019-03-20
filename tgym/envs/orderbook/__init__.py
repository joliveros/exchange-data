from abc import ABC
from tgym.envs.orderbook._long_orderbook_trading_env import \
    LongOrderBookTradingEnv
from tgym.envs.orderbook._orderbook import OrderBookTradingEnv


class OrderBookTradingEnvRLLib(OrderBookTradingEnv, ABC):
    def __init__(self, config: dict):
        OrderBookTradingEnv.__init__(self, **config)
