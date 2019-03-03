import time
from datetime import datetime
from typing import Any

import alog
import pytz

from exchange_data.bitmex_orderbook import ActionType, BitmexOrder
from exchange_data.bitmex_orderbook._action import Action
from exchange_data.emitters import TimeEmitter


class BitmexMessage(object):
    def __init__(
            self,
            table: str,
            data: Any,
            instrument_index: int,
            tick_size: float
    ):
        action_type = None
        self.action = None

        if 'action' in data:
            action_type = ActionType[data['action'].upper()]

        if 'symbol' in data:
            self.symbol: str = data['symbol']
        else:
            self.symbol = data['data'][0]['symbol']

        self.timestamp = TimeEmitter.timestamp()

        orders = [
            BitmexOrder(
                order_data,
                self.timestamp,
                instrument_index=instrument_index,
                tick_size=tick_size
            )
            for order_data in data['data']
        ]

        self.action = Action(self.symbol, table, orders, self.timestamp,
                             action_type)

    def __str__(self):
        return str(vars(self))

    @property
    def timestamp_datetime(self):
        return datetime.fromtimestamp(self.timestamp / 1000)
