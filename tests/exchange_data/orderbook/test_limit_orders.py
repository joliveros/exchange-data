import alog
import pytest

from exchange_data.orderbook import OrderBook, BuyOrder, SellOrder
from exchange_data.orderbook._order import OrderType
from exchange_data.orderbook._transcation import Transaction


@pytest.fixture
def orders():
    return [
        SellOrder(
            price=120.00,
            quantity=5
        ),
        SellOrder(
            price=110.00,
            quantity=5
        ),
        SellOrder(
            price=100.00,
            quantity=5
        ),
        BuyOrder(
            price=90.00,
            quantity=5
        ),
        BuyOrder(
            price=80.00,
            quantity=5
        ),
        BuyOrder(
            price=70.00,
            quantity=5
        )
    ]


@pytest.fixture
def orderbook(orders):
    _orderbook = OrderBook()

    for order in orders:
        _orderbook.process_order(order)

    return _orderbook


class TestOrderBook(object):
    def test_process_limit_order(self):
        order = SellOrder(
            order_type=OrderType.LIMIT,
            price=101.00,
            quantity=5
        )

        order_book = OrderBook()

        order_book.process_order(order)

    def test_crossing_limit_order(self, orderbook: OrderBook):
        crossing_limit_order = BuyOrder(101.0, 2)

        trades, order = orderbook.process_order(crossing_limit_order)

        assert order is None

        assert len(trades) == 1
        trade: Transaction = trades[0]

        # Ensure that the corresponding price level does not exist
        with pytest.raises(KeyError):
            orderbook.asks.get_price_list(crossing_limit_order.price)

        alog.debug(trade)

        order_list = orderbook.asks.price_map[100]

        assert order_list.volume == 3

        # assert trade.party1.counter_party_id == orderbook.asks.get_price_list()

# # Add orders to order book
# for order in limit_orders:
#     trades, order_id = order_book.process_order(order, False, False)
#
# # The current book may be viewed using a print
# print(order_book)
#
# # Submitting a limit order that crosses the opposing best price will result in a trade
# crossing_limit_order = {'type': 'limit',
#                         'side': 'bid',
#                         'quantity': 2,
#                         'price': 102,
#                         'trade_id': 109}
#
# print(crossing_limit_order)
# trades, order_in_book = order_book.process_order(crossing_limit_order, False, False)
# print("Trade occurs as incoming bid limit crosses best ask")
# print(trades)
# print(order_book)
#
# # If a limit crosses but is only partially matched, the remaning volume will
# # be placed in the book as an outstanding order
# big_crossing_limit_order = {'type': 'limit',
#                             'side': 'bid',
#                             'quantity': 50,
#                             'price': 102,
#                             'trade_id': 110}
# print(big_crossing_limit_order)
# trades, order_in_book = order_book.process_order(big_crossing_limit_order, False, False)
# print("Large incoming bid limit crosses best ask. Remaining volume is placed in book.")
# print(trades)
# print(order_book)
#
#
# # Market Orders
#
# # Market orders only require that a user specifies a side (bid or ask), a quantity, and their unique trade id
# market_order = {'type': 'market',
#                 'side': 'ask',
#                 'quantity': 40,
#                 'trade_id': 111}
# trades, order_id = order_book.process_order(market_order, False, False)
# print("A market order takes the specified volume from the inside of the book, regardless of price")
# print("A market ask for 40 results in:")
# print(order_book)
