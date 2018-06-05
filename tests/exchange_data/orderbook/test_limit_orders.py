from .fixtures import orderbook, orders
from exchange_data.orderbook import OrderBook, BuyOrder, SellOrder, \
    OrderType, OrderList, TradeSummary, Trade


class TestLimitOrders(object):

    def test_limit_order_cross_best_ask(self, orderbook: OrderBook):
        crossing_limit_order = BuyOrder(2, 101.0)
        trade_summary = orderbook.process_order(crossing_limit_order)

        assert len(trade_summary.trades) == 1

        trade: Trade = trade_summary.trades[0]

        min_price_list: OrderList = orderbook.asks.min_price_list()

        assert min_price_list.volume == 3
        assert trade.party1.counter_party_id == min_price_list.head_order.uid

    def test_limit_order_cross_two_levels_into_ask(self, orderbook: OrderBook):
        crossing_limit_order = BuyOrder(10, 110.0)
        trade_summary = orderbook.process_order(crossing_limit_order)

        assert len(trade_summary.trades) == 2

        min_price_list: OrderList = orderbook.asks.min_price_list()

        assert min_price_list.head_order.uid == 0

    def test_limit_order_cross_best_bid(self, orderbook: OrderBook):
        crossing_limit_order = SellOrder(2, 89.0)
        trade_summary = orderbook.process_order(crossing_limit_order)

        assert len(trade_summary.trades) == 1

        trade: TradeSummary = trade_summary.trades[0]

        max_price_list: OrderList = orderbook.bids.max_price_list()

        assert max_price_list.volume == 3
        assert trade.party1.counter_party_id == max_price_list.head_order.uid

    def test_limit_order_cross_two_levels_into_bid(self, orderbook: OrderBook):
        crossing_limit_order = SellOrder(10, 80.0)
        trade_summary = orderbook.process_order(crossing_limit_order)

        assert len(trade_summary.trades) == 2

        max_price_list: OrderList = orderbook.bids.max_price_list()

        assert max_price_list.head_order.uid == 5
