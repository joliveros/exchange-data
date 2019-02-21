from exchange_data.orderbook import Order, OrderBookSide, OrderType


class BitmexOrder(Order):
    def __init__(
            self,
            order_data: dict,
            timestamp: float,
            instrument_index: int,
            tick_size: float
    ):
        self.tick_size = tick_size
        self.instrument_index = instrument_index

        if order_data.get('size', None) is None:
            order_data['size'] = 0

        uid = order_data.get('id', None)
        quantity = order_data['size']
        price = order_data.get('price', None)

        if price:
            price = float(price)
        else:
            price = self.parse_price_from_id(uid)

        side = OrderBookSide.ASK if order_data['side'] == 'Sell' else \
            OrderBookSide.BID

        super().__init__(order_type=OrderType.LIMIT, quantity=quantity,
                         side=side, price=price, timestamp=timestamp, uid=uid)

    def parse_price_from_id(self, uid: int):
        index = 100000000 * self.instrument_index
        return float(index - uid) / 100.0
