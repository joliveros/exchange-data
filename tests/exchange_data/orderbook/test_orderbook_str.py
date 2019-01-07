import pickle

import alog

from .fixtures import orderbook, orders
from exchange_data.orderbook import OrderBook, BuyOrder


class TestOrderBookStrRepresentation(object):

    def test_orderbook_to_string(self, orderbook: OrderBook):
        trade_summary = orderbook.process_order(BuyOrder(2, 101.0))

        assert len(trade_summary.trades) == 1

        orderbook_summary = str(orderbook)
        assert len(orderbook_summary.split()) == 24
