from gym.envs.registration import register
from tgym.envs.orderbook._orderbook_frame_env import OrderBookFrameEnv


register(
    id='orderbook-trading-v0',
    entry_point='tgym.envs:OrderBookTradingEnv',
)

register(
    id='long-orderbook-trading-v0',
    entry_point='tgym.envs:LongOrderBookTradingEnv',
)


register(
    id='orderbook-frame-env-v0',
    entry_point='tgym.envs:OrderBookFrameEnv',
)
