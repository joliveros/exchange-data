from typing import List

from exchange_data.bitmex_orderbook import BitmexOrder, ActionType


class Action(object):
    def __init__(
            self,
            symbol: str,
            table: str,
            orders: List[BitmexOrder],
            timestamp: int = None,
            action_type: ActionType = None
    ):
        self.orders = orders
        self.symbol = symbol
        self.table = table
        self.timestamp = timestamp
        self.type = action_type

    def __str__(self):
        return str(vars(self))