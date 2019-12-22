from exchange_data.utils import EventEmitterBase
from exchange_data.bitmex_orderbook import ActionType, BitmexOrder
from exchange_data.bitmex_orderbook._bitmex_message import BitmexMessage
from exchange_data.bitmex_orderbook._instrument_info import InstrumentInfo
from exchange_data.channels import BitmexChannels
from exchange_data.orderbook import OrderBook, OrderType
from exchange_data.orderbook.exceptions import PriceDoesNotExistException
from typing import Optional

import alog


class BitmexOrderBook(OrderBook, EventEmitterBase):

    def __init__(self, symbol: BitmexChannels, **kwargs):
        super().__init__(symbol=symbol, **kwargs)

        instrument_info = InstrumentInfo.get_instrument(symbol.value)

        self.__dict__.update(instrument_info.__dict__)

        self.symbol = symbol

        self.on('orderBookL2', self.message)

    def message(self, raw_message) -> Optional[BitmexMessage]:
        if not isinstance(raw_message, dict):
            raise Exception()

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

            try:
                self.order_book_l2(message)
            except PriceDoesNotExistException:
                pass
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

