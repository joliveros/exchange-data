import alog

from .fixtures import orderbook, orders
from exchange_data.orderbook import OrderBook, SellMarketOrder, \
    BuyMarketOrder, OrderList, TradeSummary, Trade


class TestMarketOrders(object):

    def test_buy_market_order_one_price_level(self, orderbook: OrderBook):
        market_order = BuyMarketOrder(2)
        trade_summary = orderbook.process_order(market_order)

        assert trade_summary.order is None
        assert len(trade_summary.trades) == 1

        trade: Trade = trade_summary.trades[0]

        min_price_list: OrderList = orderbook.asks.min_price_list()

        assert min_price_list.volume == 3
        assert trade.party1.counter_party_id == min_price_list.head_order.uid

    def test_buy_market_order_two_price_levels(self, orderbook: OrderBook):
        market_order = BuyMarketOrder(10)
        trade_summary = orderbook.process_order(market_order)

        assert trade_summary.order is None
        assert len(trade_summary.trades) == 2

        assert orderbook.asks.volume == 5

    def test_buy_market_order_exceeds_all_levels(self, orderbook: OrderBook):
        market_order = BuyMarketOrder(16)
        trade_summary = orderbook.process_order(market_order)

        assert orderbook.bids.volume == 15
        assert orderbook.asks.volume == 0

        assert trade_summary.quantity_to_trade == 1
