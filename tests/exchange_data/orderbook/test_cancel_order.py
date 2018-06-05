import pytest

from .fixtures import orderbook, orders
from exchange_data.orderbook import OrderBook, BuyOrder
from exchange_data.orderbook import OrderExistsException


class TestCancelOrder(object):

    def test_cancel_bid_order(self, orderbook: OrderBook):
        market_order = BuyOrder(2, 90.0)

        orderbook.process_order(market_order)

        orderbook.cancel_order(market_order.uid)

        assert orderbook.bids.order_exists(market_order) is False

    def test_cancel_non_existant_order_raises_exception(self,
                                                        orderbook: OrderBook):
        with pytest.raises(OrderExistsException):
            orderbook.cancel_order(10)
