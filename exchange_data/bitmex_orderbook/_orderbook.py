from dateutil.tz import tz

from exchange_data.bitmex_orderbook import ActionType, BitmexOrder
from exchange_data.bitmex_orderbook._bitmex_message import BitmexMessage
from exchange_data.bitmex_orderbook._instrument_info import InstrumentInfo
from exchange_data.channels import BitmexChannels
from exchange_data.orderbook import OrderBook, OrderType
from exchange_data.orderbook.exceptions import PriceDoesNotExistException
from pyee import EventEmitter
from typing import Optional
from datetime import datetime

import alog


class BitmexOrderBook(OrderBook, EventEmitter):

    def __init__(self, symbol: BitmexChannels):
        OrderBook.__init__(self)
        EventEmitter.__init__(self)

        instrument_info = InstrumentInfo.get_instrument(symbol.value)

        self.__dict__.update(instrument_info.__dict__)

        self.symbol = symbol.value
        self.on('orderBookL2', self.message)

    def message(self, raw_message) -> Optional[BitmexMessage]:
        if not isinstance(raw_message, dict):
            raise Exception()

        expected_keys = ['action', 'data', 'table', 'symbol']

        # for key in raw_message.keys():
        #     assert key in expected_keys

        message = None

        table = raw_message['table']

        if table in ['orderBookL2', 'trade']:
            order_type = None
            if table == 'orderBookL2':
                order_type = OrderType.LIMIT
            elif table == 'trade':
                order_type = OrderType.MARKET

            message = BitmexMessage(
                table,
                raw_message,
                instrument_index=self.index,
                tick_size=self.tick_size,
                order_type=order_type
            )
            self.last_timestamp = message.timestamp

            self.order_book_l2(message)
        else:
            raise Exception(table)

        return message

    def order_book_l2(self, message: BitmexMessage):
        if message.action.type.value == ActionType.UPDATE.value:
            self.update_orders(message)

        elif message.action.type.value in [
            ActionType.INSERT.value, ActionType.PARTIAL.value
        ]:
            for order in message.action.orders:
                self.process_order(order)

        elif message.action.type.value == ActionType.DELETE.value:
            order: BitmexOrder = None
            for order in message.action.orders:
                try:
                    self.cancel_order(order.uid)
                except Exception:
                    pass

    def update_orders(self, message: BitmexMessage):
        orders = message.action.orders

        for order in orders:
            uid = order.uid

            if self.order_exists(uid):
                try:
                    self.modify_order(order.uid, order.price,
                                      quantity=order.quantity,
                                      timestamp=order.timestamp)
                except PriceDoesNotExistException:
                    pass
            else:
                self.process_order(order)

    def relative_orderbook(self):
        if self.bids is not None and len(self.bids) > 0:
            alog.debug(self.bids.price_tree.max_key())

        if self.asks is not None and len(self.asks) > 0:
            alog.debug(self.asks.price_tree.min_key())

