import json
from enum import Enum

import alog
from stringcase import snakecase

from exchange_data.orderbook._order import Order
from ._tf_orderbook import TFLimitOrderBook


class Action(Enum):
    INSERT = 0
    UPDATE = 1
    DELETE = 2


class BitmexOrder(Order):
    def __init__(self, order_data: dict, timestamp):

        if order_data.get('size', None) is None:
            order_data['size'] = 0

        uid = order_data.get('id', None)

        super().__init__(uid, order_data['side'] == 'Buy',
                         order_data['size'], order_data.get('price', None),
                         timestamp=timestamp)


class BitmexMessage(object):
    def __init__(self, data: dict):
        self.timestamp = None
        self.data = []
        self.__dict__ = data

        self.action = Action[self.action.upper()]

        if type(self.data) is list:
            self.data = [BitmexOrder(order_data, self.timestamp) for order_data
                         in self.data]
        else:
            self.data = [BitmexOrder(self.data, self.timestamp)]

    def __str__(self):
        return str(self.__dict__)


class TFBitmexLimitOrderBook(TFLimitOrderBook):
    def __init__(self, symbol: str, total_time='1d', save_json=False,
                 json_file=None):
        TFLimitOrderBook.__init__(self, total_time=total_time,
                                  database='bitmex', symbol=symbol,
                                  save_json=save_json, json_file=json_file)

    def on_message(self, raw_message):
        data = json.loads(raw_message['data'])
        data['timestamp'] = raw_message['time']
        table = data['table']

        getattr(self, snakecase(table))(data)

    def order_book_l2(self, data):
        message = BitmexMessage(data)

        self.process_order(data)

    def quote(self, data):
        pass

    def trade(self, data):
        for order in BitmexMessage(data).data:
            super().trade(order)
