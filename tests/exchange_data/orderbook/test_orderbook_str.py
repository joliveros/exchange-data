import pickle

import alog

from .fixtures import orderbook, orders
from exchange_data.orderbook import OrderBook, BuyOrder


class TestOrderBookStrRepresentation(object):

    def test_orderbook_to_string(self, orderbook: OrderBook):
        trade_summary = orderbook.process_order(BuyOrder(2, 101.0))

        assert len(trade_summary.trades) == 1

        orderbook_summary = str(orderbook)
        assert len(orderbook_summary.split()) == 40

    def test_(self, orderbook: OrderBook):
        # alog.info(orderbook)
        # orderbook_data = pickle.dumps(orderbook)
        # alog.info(orderbook_data)
        #
        # _orderbook = pickle.loads(orderbook_data)
        # alog.info(alog.pformat(vars(_orderbook)))
        # alog.info(_orderbook)

        alog.info(OrderBook)
