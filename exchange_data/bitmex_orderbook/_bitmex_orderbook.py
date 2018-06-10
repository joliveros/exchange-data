import json
from enum import auto
from typing import Any

from exchange_data.orderbook import Order, NoValue, OrderBookSide, OrderType
from exchange_data._orderbook import InfluxOrderBook


class ActionType(NoValue):
    INSERT = auto()
    UPDATE = auto()
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
    def __init__(self, symbol: str, data: str):
        self.symbol = symbol
        data = json.loads(data)
        self.type = ActionType[data['action'].upper()]
        self.table = data['table']
        self.timestamp = data['timestamp']
        self.orders = data['data']


class BitmexMessage(object):
    def __init__(self, data: Any):
        if type(data) == str:
            data = json.loads(data)
        self.data: Action = None
        self.symbol: str = data['symbol']
        self.time: int = data['time']

        self.action = Action(self.symbol, data['data'])

    def __str__(self):
        return str(self.__dict__)


class BitmexOrderBook(InfluxOrderBook):
    def __init__(self, symbol: str, total_time='1d', save_json=False,
                 json_file=None):
        InfluxOrderBook.__init__(self, total_time=total_time,
                                 database='bitmex', symbol=symbol,
                                 save_json=save_json, json_file=json_file)

    def on_message(self, raw_message):
        message = BitmexMessage(raw_message)

        if message.action.table == 'orderBookL2':
            self.order_book_l2(message)

    def order_book_l2(self, message):
        if message.action.type == ActionType.UPDATE:
            self.update_orders(message.action.orders)

        elif message.action.type == ActionType.INSERT:
            orders = [BitmexOrder(order_data, message.time) for order_data in
                      message.action.orders]

            for order in orders:
                self.process_order(order)

        elif message.action.type == ActionType.DELETE:
            for order_data in message.action.orders:
                self.cancel_order(order_data['id'])

    def update_orders(self, orders):
        for order in orders:
            self.modify_order(order['id'], price=None, quantity=order['size'])
