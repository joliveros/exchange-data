from ._orderbook_trading import OrderBookTradingEnv
from ._trading import SpreadTrading
from gym.envs.registration import register


register(
    id='orderbook-trading-v0',
    entry_point='tgym.envs:OrderBookTradingEnv',
)
