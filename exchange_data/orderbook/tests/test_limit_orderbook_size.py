import alog

from .fixtures import orderbook, orders
from exchange_data.orderbook import OrderBook, BuyOrder, SellOrder


class TestSizeLimit(object):

    def test_limit_order_new_level(self, orderbook: OrderBook):
        orderbook.max_depth = 3
        crossing_limit_order = BuyOrder(10, 79.99)
        orderbook.process_order(crossing_limit_order)
        orderbook.process_order(BuyOrder(10, 90.00))
        orderbook.process_order(BuyOrder(10, 69.00))
        orderbook.process_order(SellOrder(10, 121.00))
        orderbook.process_order(SellOrder(10, 123.00))
        orderbook.process_order(SellOrder(10, 120.00))

        assert len(orderbook.bids.price_map) == orderbook.max_depth
        assert len(orderbook.asks.price_map) == orderbook.max_depth
