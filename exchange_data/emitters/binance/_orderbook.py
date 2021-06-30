from exchange_data.orderbook import OrderBook, OrderType, OrderBookSide, Order
from exchange_data.orderbook.exceptions import PriceDoesNotExistException
from exchange_data.utils import EventEmitterBase, DateTimeUtils


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
            price_list, s = self.get_price(price)

            if s == OrderBookSide.ASK:
                self.asks.remove_order_by_id(price_list.head_order.uid)
            else:
                self.bids.remove_order_by_id(price_list.head_order.uid)

            remaining_vol = self.get_volume(price)

            if remaining_vol > 0.0:
                self.remove_price(price, side, timestamp)

        except PriceDoesNotExistException as e:
            pass

    def update_price(self, price, quantity, side, timestamp):
        try:
            current_quantity = self.get_volume(price)
        except PriceDoesNotExistException as e:
            current_quantity = 0.0

        if current_quantity != quantity:
            self.remove_price(price, side, timestamp)

            self.process_order(Order(
                order_type=OrderType.LIMIT,
                price=price,
                quantity=quantity,
                side=side,
                timestamp=timestamp
            ))

