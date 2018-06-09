from .fixtures import orderbook, orders
from exchange_data.orderbook import OrderBook, BuyOrder


class TestOrderBookStrRepresentation(object):

    def test_(self, orderbook: OrderBook):
        trade_summary = orderbook.process_order(BuyOrder(2, 101.0))

        assert len(trade_summary.trades) == 1

        orderbook_summary = orderbook.__str__()
        assert len(orderbook_summary.split()) == 40



