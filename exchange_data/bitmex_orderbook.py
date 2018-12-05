import pdb
from datetime import datetime
from enum import auto, Enum
from functools import lru_cache
from typing import Any, List

import alog
import json
import time
import traceback

import requests

from exchange_data.orderbook import Order, NoValue, OrderBookSide, OrderType, \
    OrderBook
from exchange_data.orderbook.exceptions import OrderExistsException, \
    PriceDoesNotExistException


class InstrumentInfo(object):
    INSTRUMENTS_URL = 'https://www.bitmex.com/api/v1/instrument?columns' \
                      '=symbol,tickSize&start=0&count=500'

    def __init__(
            self,
            index: int,
            symbol: str,
            tick_size: float,
            timestamp: str
    ):
        self.index = index
        self.timestamp = timestamp
        self.tick_size = tick_size
        self.symbol = symbol

    @staticmethod
    def get_instrument(symbol: str) -> 'InstrumentInfo':
        if not symbol.isupper():
            raise Exception('symbol should be uppercase.')

        r = requests.get(InstrumentInfo.INSTRUMENTS_URL)
        all_instruments = r.json()

        data = [data for data in all_instruments if data['symbol'] == symbol][0]

        index = all_instruments.index(data)

        return InstrumentInfo(
            index=index,
            symbol=symbol,
            tick_size=data['tickSize'],
            timestamp=data['timestamp']
        )


class ActionType(NoValue):
    INSERT = auto()
    UPDATE = auto()
    PARTIAL = auto()
    DELETE = auto()


class BitmexOrder(Order):
    def __init__(self, order_data: dict, timestamp: int):
        if order_data.get('size', None) is None:
            order_data['size'] = 0

        uid = order_data.get('id', None)
        quantity = order_data['size']
        price = order_data.get('price', None)

        if price:
            price = float(price)

        side = OrderBookSide.ASK if order_data['side'] == 'Sell' else \
            OrderBookSide.BID

        super().__init__(order_type=OrderType.LIMIT, quantity=quantity,
                         side=side, price=price, timestamp=timestamp, uid=uid)


class Action(object):
    def __init__(
            self,
            symbol: str,
            table: str,
            orders: List[BitmexOrder],
            timestamp: int=None,
            action_type: ActionType = None
    ):
        self.orders = orders
        self.symbol = symbol
        self.table = table
        self.timestamp = timestamp
        self.type = action_type

    def __str__(self):
        return str(vars(self))


class NotOrderbookMessage(Exception):
    pass


class BitmexMessage(object):
    def __init__(self, data: Any):
        if isinstance(data, str):
            data = json.loads(data)

        try:
            _data = data['data']

            if isinstance(_data, str):
                data = json.loads(_data)

        except KeyError:
            pass
        except TypeError:
            pass

        try:
            table = data['table']
        except:
            pdb.set_trace()

        if table != 'orderBookL2':
            raise NotOrderbookMessage('Not orderBookL2 message')

        action_type = None
        self.action = None

        if 'action' in data:
            action_type = ActionType[data['action'].upper()]

        if 'symbol' in data:
            self.symbol: str = data['symbol']
        else:
            self.symbol = data['data'][0]['symbol']

        if 'time' in data:
            self.timestamp: int = data['time']
        else:
            self.timestamp = time.time()

        orders = [BitmexOrder(order_data, self.timestamp)
                  for order_data in data['data']]
        self.action = Action(self.symbol, table, orders, self.timestamp, action_type)

    def __str__(self):
        return str(vars(self))

    @property
    def timestamp_datetime(self):
        return datetime.fromtimestamp(self.timestamp / 1000)


class BitmexTickSize(Enum):
    XBTUSD = 0.01


class BitmexOrderBook(OrderBook):

    def __init__(self, symbol: str):
        OrderBook.__init__(self)
        instrument_info = InstrumentInfo.get_instrument(symbol)
        self.__dict__.update(instrument_info.__dict__)

        self.symbol = symbol
        self.result_set = None
        self.last_timestamp = None

    def message(self, raw_message) -> BitmexMessage:
        message = BitmexMessage(raw_message)

        self.last_timestamp = message.timestamp

        if message.action is None:
            return message

        if message.action.table == 'orderBookL2':
            self.order_book_l2(message)

        elif message.action.table == 'trade':
            pass

        alog.info(message)

        return message


    def order_book_l2(self, message: BitmexMessage):

        if message.action.type == ActionType.UPDATE:
            self.update_orders(message)

        elif message.action.type in [ActionType.INSERT, ActionType.PARTIAL]:
            for order in message.action.orders:
                self.process_order(order)

        elif message.action.type == ActionType.DELETE:
            for order_data in message.action.orders:
                try:
                    self.cancel_order(order_data['id'])
                except Exception:
                    pass

    def update_orders(self, message: BitmexMessage):
        orders = message.action.orders
        timestamp = message.timestamp

        for order in orders:
            try:
                uid = order['id']

                if 'price' not in order:
                    order['price'] = self.parse_price_from_id(uid)

                if self.order_exists(uid):
                    self.modify_order(order['id'], order['price'],
                                      quantity=order['size'],
                                      timestamp=timestamp)
                else:
                    new_order = BitmexOrder(order, message.timestamp)
                    self.process_order(new_order)

            except Exception as e:
                pass

    def relative_orderbook(self):
        if self.bids is not None and len(self.bids) > 0:
            alog.debug(self.bids.price_tree.max_key())

        if self.asks is not None and len(self.asks) > 0:
            alog.debug(self.asks.price_tree.min_key())

    @lru_cache(maxsize=None)
    def parse_price_from_id(self, id: int):
        return ((100000000 * self.index) - id) * self.tick_size

