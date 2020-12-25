from exchange_data.bitmex_orderbook._action import Action
from exchange_data.emitters import TimeEmitter
from exchange_data.utils import EventEmitterBase, DateTimeUtils
from exchange_data.bitmex_orderbook import ActionType, BitmexOrder
from exchange_data.bitmex_orderbook._instrument_info import InstrumentInfo
from exchange_data.channels import BitmexChannels
from exchange_data.orderbook import OrderBook, OrderType, OrderBookSide, Order
from exchange_data.orderbook.exceptions import PriceDoesNotExistException
from typing import Optional, Any, List

import alog



class BinanceOrderBook(OrderBook, EventEmitterBase):

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol=symbol, **kwargs)

        self.symbol = symbol

        self.on(self.symbol, self.message)

    def message(self, raw_message):
        asks = raw_message['a']
        bids = raw_message['b']
        timestamp = DateTimeUtils.parse_db_timestamp(raw_message['E'])

        levels = [(float(price), float(quantity), OrderBookSide.ASK)
                  for price, quantity in asks]
        levels += [(float(price), float(quantity), OrderBookSide.BID)
                   for price, quantity in bids]

        for price, quantity, side in levels:
            if quantity == 0.0:
                self.remove_price(price, side, timestamp)
            else:
                self.update_price(price, quantity, side, timestamp)

    def remove_price(self, price, side, timestamp):
        try:
            current_quantity = self.get_volume(price)

            if current_quantity > 0.0:
                self.process_order(Order(
                    order_type=OrderType.MARKET,
                    price=price,
                    quantity=current_quantity,
                    side=side,
                    timestamp=timestamp
                ))

        except PriceDoesNotExistException as e:
            pass

    def update_price(self, price, quantity, side, timestamp):
        try:
            self.process_order(Order(
                order_type=OrderType.LIMIT,
                price=price,
                quantity=quantity,
                side=side,
                timestamp=timestamp
            ))

        except PriceDoesNotExistException as e:
            order = Order(
                order_type=OrderType.LIMIT,
                price=price,
                quantity=quantity,
                side=side,
                timestamp=timestamp
            )

            if side == OrderBookSide.BID:
                self.bids.insert_order(order)
            else:
                self.asks.insert_order(order)

