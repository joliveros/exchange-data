import sys
from datetime import datetime
from enum import auto, Enum
from functools import lru_cache
from typing import Any

import alog
import json
import time
import traceback

import requests

from exchange_data.orderbook import Order, NoValue, OrderBookSide, OrderType, \
    OrderBook
from exchange_data.orderbook.exceptions import OrderExistsException, \
    PriceDoesNotExistException


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
    def __init__(self, symbol: str, data: str, timestamp: int=None):
        self.symbol = symbol

        if type(data) == str:
            data = json.loads(data)

        self.type = ActionType[data['action'].upper()]
        self.table = data['table']

        if 'timestamp' in data:
            self.timestamp = data['timestamp']
        elif timestamp:
            self.timestamp = timestamp
        else:
            self.timestamp = time.time()

        self.orders = data['data']

    def __str__(self):
        return self.__dict__


class BitmexMessage(object):
    def __init__(self, data: Any):

        if type(data) == str:
            data = json.loads(data)

        if 'symbol' in data:
            self.symbol: str = data['symbol']
        else:
            self.symbol = data['data'][0]['symbol']

        if 'time' in data:
            self.timestamp: int = data['time']
        else:
            self.timestamp = time.time()

        if 'data' in data:
            data = data['data']

        self.action = Action(self.symbol, data, self.timestamp)

    def __str__(self):
        return str(self.__dict__)

    @property
    def timestamp_datetime(self):
        return datetime.fromtimestamp(self.timestamp / 1000)


class BitmexTickSize(Enum):
    XBTUSD = 0.01


class BitmexOrderBook(OrderBook):

    def __init__(self, symbol: str):
        OrderBook.__init__(self)

        self.symbol = symbol
        self.result_set = None
        self.last_timestamp = None
        self._get_instrument_info()

    def message_strict(self, raw_message):
        message = BitmexMessage(raw_message)
        self.last_timestamp = message.timestamp

        if message.action.table == 'orderBookL2':
            self.order_book_l2(message)

        elif message.action.table == 'trade':
            pass

        return message

    def message(self, raw_message) -> BitmexMessage:
        try:
            message = BitmexMessage(raw_message)
            self.last_timestamp = message.timestamp

            if message.action.table == 'orderBookL2':
                self.order_book_l2(message)

            elif message.action.table == 'trade':
                pass

            return message

        except Exception:
            traceback.print_exc()

    def order_book_l2(self, message):

        if message.action.type == ActionType.UPDATE:
            self.update_orders(message)

        elif message.action.type in [ActionType.INSERT, ActionType.PARTIAL]:
            orders = [BitmexOrder(order_data, message.timestamp)
                      for order_data in message.action.orders]

            for order in orders:
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



    def _get_instrument_info(self):
        all_instruments = self._instrument_data()
        instrument_data = [data for data in all_instruments if data['symbol'] == self.symbol][0]

        self.index = all_instruments.index(instrument_data)

        self.tick_size = instrument_data['tickSize']

        if BitmexTickSize[self.symbol]:
            self.tick_size = BitmexTickSize[self.symbol].value


class InstrumentInfo(object):


    def __init__(
        self,
        symbol: str,
        tickSize: float,
        timestamp: str,
        index: int
    ):
        self.symbol = symbol

    @staticmethod
    def get_instrument(symbol: str):
        INSTRUMENTS_URL = 'https://www.bitmex.com/api/v1/instrument?columns' \
                  '=symbol,tickSize&start=0&count=500'

        r = requests.get(INSTRUMENTS_URL)
        all_instruments = r.json()

        data = [data for data in all_instruments if data['symbol'] == symbol][0]

        index = all_instruments.index(data)

        return InstrumentInfo(symbol, tickSize=data['tickSize'], timestamp=data['timestamp'], index=index)

