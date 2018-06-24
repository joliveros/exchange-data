from .fixtures import orderbook, orders
from exchange_data.orderbook import OrderBook, BuyOrder, SellOrder


class TestModifyOrder(object):

    def test_modify_buy_order_price(self, orderbook: OrderBook):
        order = BuyOrder(2.0, 90.0)

        orderbook.process_order(order)

        assert orderbook.bids.price_map[90.0].volume == 7

        orderbook.modify_order(order.uid, 91.0, 2.0)

        assert orderbook.bids.price_map[90.0].volume == 5
        assert orderbook.bids.price_map[91.0].volume == 2

    def test_modify_sell_order_price(self, orderbook: OrderBook):
        order = SellOrder(2.0, 120.0)

        orderbook.process_order(order)

        assert orderbook.asks.price_map[120.0].volume == 7

        orderbook.modify_order(order.uid, 121.0, 2.0)

        assert orderbook.asks.price_map[120.0].volume == 5
        assert orderbook.asks.price_map[121.0].volume == 2

    def test_increase_sell_order_quantity(self, orderbook: OrderBook):
        order = SellOrder(2.0, 120.0)

        orderbook.process_order(order)

        assert orderbook.asks.volume == 17.0

        orderbook.modify_order(order.uid, order.price, 3)

        assert orderbook.asks.volume == 18.0

    def test_decrease_sell_order_quantity(self, orderbook: OrderBook):
        order = SellOrder(2.0, 120.0)

        orderbook.process_order(order)

        assert orderbook.asks.volume == 17.0

        orderbook.modify_order(order.uid, order.price, 1.0)

        assert orderbook.asks.volume == 16.0

    def test_increase_buy_order_quantity(self, orderbook: OrderBook):
        order = BuyOrder(2.0, 90.0)

        orderbook.process_order(order)

        assert orderbook.bids.volume == 17.0

        orderbook.modify_order(order.uid, order.price, 3)

        assert orderbook.bids.volume == 18.0

    def test_decrease_buy_order_quantity(self, orderbook: OrderBook):
        order = BuyOrder(2.0, 90.0)

        orderbook.process_order(order)

        assert orderbook.bids.volume == 17.0

        orderbook.modify_order(order.uid, order.price, 1)

        assert orderbook.bids.volume == 16.0
