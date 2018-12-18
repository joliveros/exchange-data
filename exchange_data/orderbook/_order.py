import time
from enum import Enum, auto
from exchange_data.orderbook._orderlist import OrderList


class NoValue(Enum):
    def __repr__(self):
        return '<%s.%s>' % (self.__class__.__name__, self.name)


class OrderType(NoValue):
    LIMIT = 0,
    MARKET = 1


class OrderBookSide(NoValue):
    ASK = 0,
    BID = 1


class InvalidOrderQuantity(BaseException):
    pass


class InvalidPriceMarketOrderException(Exception):
    pass


class PriceMustBeFloatException(Exception):
    pass


class Order(object):
    """
    Orders represent the core piece of the exchange. Every bid/ask is an Order.
    Orders are doubly linked and have helper functions (next_order, prev_order)
    to help the exchange fulfill orders with quantities larger than a single
    existing Order.
    """
    next_order: 'Order'
    order_list: OrderList
    prev_order: 'Order'

    def __init__(self,
                 order_type: OrderType,
                 quantity: float,
                 side: OrderBookSide,
                 price: float=None,
                 timestamp: int = None,
                 uid: int = None
                 ):
        self.next_order = None
        self.order_list = None
        self.prev_order = None

        if price:
            if type(price) != float:
                raise PriceMustBeFloatException()

        self.price = price
        self.side = side
        self._timestamp = timestamp
        self.type = order_type
        self.uid = uid

        if quantity < 0:
            raise InvalidOrderQuantity(quantity)
        else:
            self.quantity = quantity

        if self.type == OrderType and self.price:
            raise InvalidPriceMarketOrderException()

    @property
    def timestamp(self):
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value):
        self._timestamp = value

    def update_quantity(self, new_quantity):
        new_timestamp = time.time()
        if new_quantity > self.quantity and self.order_list.tail_order != self:
            # check to see that the order is not the last order in list and
            # the quantity is more
            self.order_list.move_to_tail(self)  # move to the end

        # update volume
        self.order_list.volume -= (self.quantity - new_quantity)
        self.timestamp = new_timestamp
        self.quantity = new_quantity

    def __str__(self):
        return f'{self.quantity}@{self.price}/{self.uid} - {self.timestamp}'


class BuyOrder(Order):
    def __init__(self,
                 quantity: float,
                 price: float=None,
                 order_type: OrderType = OrderType.LIMIT
                 ):
        super().__init__(
            order_type=order_type,
            price=price,
            quantity=quantity,
            side=OrderBookSide.BID
        )


class SellOrder(Order):
    def __init__(self,
                 quantity: float,
                 price: float=None,
                 order_type: OrderType = OrderType.LIMIT
                 ):
        super().__init__(
            order_type=order_type,
            price=price,
            quantity=quantity,
            side=OrderBookSide.ASK
        )


class BuyMarketOrder(BuyOrder):
    def __init__(self, quantity: float):
        super().__init__(order_type=OrderType.MARKET, quantity=quantity)


class SellMarketOrder(SellOrder):
    def __init__(self, quantity: float):
        super().__init__(order_type=OrderType.MARKET, quantity=quantity)
