from gym.envs.registration import register
from tgym.envs.orderbook import OrderBookTradingEnv, LongOrderBookTradingEnv


register(
    id='orderbook-trading-v0',
    entry_point='tgym.envs:OrderBookTradingEnv',
)

register(
    id='long-orderbook-trading-v0',
    entry_point='tgym.envs:LongOrderBookTradingEnv',
)
