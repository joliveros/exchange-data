import json
from enum import Enum

import alog

from exchange_data.limit_orderbook import Order
from exchange_data.limit_orderbook._limit_level import LimitLevelBalanceError
from exchange_data.utils import nice_print
from ._tf_orderbook import TFLimitOrderBook


class Action(Enum):
    INSERT = 0
    UPDATE = 1
    DELETE = 2


class BitmexOrder(Order):
    def __init__(self, order_data: dict, timestamp):
        if order_data.get('size', None) is None:
            order_data['size'] = 0

        Order.__init__(self, order_data['id'], order_data['side'] == 'Buy', order_data['size'],
                       order_data.get('price', None), timestamp=timestamp)


class Message(object):
    def __init__(self, _json: str, timestamp: int):
        self.timestamp = timestamp
        self.__dict__ = json.loads(_json)
        self.action = Action[self.action.upper()]

        if type(self.data) is list:
            self.data = [BitmexOrder(order_data, timestamp) for order_data in self.data]
        else:
            self.data = [BitmexOrder(self.data, timestamp)]

    def __str__(self):
        return str(self.__dict__)


class TFBitmexLimitOrderBook(TFLimitOrderBook):
    def __init__(self, symbol: str, total_time='1d', save_json=False, json_file=None):
        TFLimitOrderBook.__init__(self, total_time=total_time, database='bitmex', symbol=symbol,
                                  save_json=save_json, json_file=json_file)

    def on_message(self, raw_message):
        message = Message(raw_message['data'], raw_message['time'])

        try:
            if message.action == Action.INSERT:
                for order in message.data:
                    self.add(order)
            elif message.action == Action.UPDATE:
                for order in message.data:
                    self.update(order)
            elif message.action == Action.DELETE:
                for order in message.data:
                    self.remove(order)

        except KeyError as e:
            # print(e)
            pass
        except LimitLevelBalanceError as e:
            print(e)
            pass
