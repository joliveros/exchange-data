from enum import Enum
from exchange_data.bitmex_orderbook._action_type import ActionType
from exchange_data.bitmex_orderbook._instrument_info import InstrumentInfo
from exchange_data.bitmex_orderbook._order import BitmexOrder
from exchange_data.bitmex_orderbook._bitmex_message import BitmexMessage
from exchange_data.bitmex_orderbook._orderbook import BitmexOrderBook


class NotOrderbookMessage(Exception):
    pass


class BitmexTickSize(Enum):
    XBTUSD = 0.01



