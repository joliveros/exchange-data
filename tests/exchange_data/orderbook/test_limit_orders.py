import alog

from .fixtures import orderbook, orders
from exchange_data.orderbook import OrderBook, BuyOrder, SellOrder, \
    OrderType, OrderList, Transaction


class TestLimitOrders(object):
    def test_process_limit_order(self):
        order = SellOrder(order_type=OrderType.LIMIT, price=101.00, quantity=5)

        order_book = OrderBook()

        order_book.process_order(order)

    def test_limit_order_cross_best_ask(self, orderbook: OrderBook):
        crossing_limit_order = BuyOrder(101.0, 2)
        trades, order = orderbook.process_order(crossing_limit_order)

        assert order is None
        assert len(trades) == 1

        trade: Transaction = trades[0]

        min_price_list: OrderList = orderbook.asks.min_price_list()

        assert min_price_list.volume == 3
        assert trade.party1.counter_party_id == min_price_list.head_order.uid

    def test_limit_order_cross_two_levels_into_ask(self, orderbook: OrderBook):
        crossing_limit_order = BuyOrder(110.0, 10)
        trades, order = orderbook.process_order(crossing_limit_order)

        assert order is None
        assert len(trades) == 2

        min_price_list: OrderList = orderbook.asks.min_price_list()

        assert min_price_list.head_order.uid == 0

    def test_limit_order_cross_best_bid(self, orderbook: OrderBook):
        crossing_limit_order = SellOrder(89, 2)
        trades, order = orderbook.process_order(crossing_limit_order)

        assert order is None
        assert len(trades) == 1

        trade: Transaction = trades[0]

        max_price_list: OrderList = orderbook.bids.max_price_list()

        assert max_price_list.volume == 3
        assert trade.party1.counter_party_id == max_price_list.head_order.uid

    def test_limit_order_cross_two_levels_into_bid(self, orderbook: OrderBook):
        crossing_limit_order = SellOrder(80, 10)
        trades, order = orderbook.process_order(crossing_limit_order)

        assert order is None
        assert len(trades) == 2

        max_price_list: OrderList = orderbook.bids.max_price_list()

        assert max_price_list.head_order.uid == 5

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
