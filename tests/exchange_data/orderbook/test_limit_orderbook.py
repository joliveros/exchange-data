from unittest import TestCase

import alog
import pytest

from exchange_data.limit_orderbook import LimitOrderBook, Order

@pytest.fixture
def orders():

    return [
        Order(uid=1, is_bid=True, size=5, price=101.00),
        Order(uid=2, is_bid=True, size=5, price=95.00),
        Order(uid=3, is_bid=True, size=5, price=90.00),
        Order(uid=4, is_bid=False, size=5, price=200.00),
        Order(uid=5, is_bid=False, size=5, price=205.00),
        Order(uid=6, is_bid=False, size=5, price=210.00),
    ]

@pytest.fixture
def limit_order_book(orders):
    lob = LimitOrderBook()

    for order in orders:
        lob.process(order)

    return lob


@pytest.mark.usefixtures('limit_order_book')
class TestLimitOrderBook(object):

    def test_adding_a_new_order_works(self, orders, limit_order_book):
        lob = limit_order_book

        bid_order = Order(uid=1, is_bid=True, size=10, price=101.0)
        ask_order = Order(uid=4, is_bid=False, size=10, price=200.0)

        lob.process(bid_order)
        lob.process(ask_order)

        assert lob.best_ask.price == 200
        assert lob.best_bid.price == 101

        assert lob.best_bid.volume == 1010

        # Assert that the best bid (bid_order) has no previous and no next item,
        # since it is the only one in the book on the bid size at the moment.
        assert len(lob.best_bid) == 1
        assert len(lob.best_ask) == 1

        assert bid_order.next_item is None
        assert bid_order.previous_item is None

        assert lob.best_ask.size == ask_order.size
        assert lob.best_bid.size == bid_order.size

        assert 1 in lob._orders

        # Assert that updating an order works
        updated_bid_order = Order(uid=1, is_bid=True, size=4, price=101.0,
                                  timestamp=bid_order.timestamp)
        lob.process(updated_bid_order)

        assert lob.best_bid.size == 4
        assert lob.best_bid.volume == 404

        updated_ask_order = Order(uid=4, is_bid=True, size=4, price=200.0,
                                  timestamp=ask_order.timestamp)
        lob.process(updated_ask_order)

        assert lob.best_ask.size == 4
        assert lob.best_ask.volume == 800

        # Assert that adding an additional order to a limit level updates the
        # doubly linked list correctly
        bid_order_2 = Order(is_bid=True, size=5, price=101.00)
        lob.process(bid_order_2)

        assert lob.best_bid.orders.head.next_item == bid_order_2
        assert len(lob.best_bid) == 2

    def test_removing_orders_works(self, orders, limit_order_book):
        lob = limit_order_book

        # Assert that removing an order from a limit level with several
        # orders resets the tail, head and previous / next items accordingly
        removed_bid_order = Order(uid=1, is_bid=True, size=0, price=101.0)

        assert len(lob.best_bid) == 1

        assert lob.best_bid.orders.head == orders[0]
        assert lob.best_bid.orders.tail == orders[0]

        # alog.debug(lob._price_levels)

        alog.debug(lob.bids)

        lob.process(removed_bid_order)

        # assert len(lob.best_bid) == 1
        # assert lob.best_bid.orders.head == orders[1]
    #     self.assertEqual(lob.best_bid.orders.head, bid_order_2)
    #     self.assertEqual(lob.best_bid.orders.tail, bid_order_2)
    #     self.assertIsNone(lob.best_bid.orders.head.next_item)
    #     self.assertIsNone(lob.best_bid.orders.head.previous_item)
    #     self.assertNotIn(removed_bid_order.uid, lob._orders)
    #     self.assertIn(removed_bid_order.price, lob._price_levels)
    #
    #     # Assert that removing the last Order in a price level removes its
    #     # limit Level accordingly
    #     removed_bid_order_2 = Order(uid=2, is_bid=True, size=0, price=100)
    #     lob.process(removed_bid_order_2)
    #     self.assertIsNone(lob.best_bid)
    #
    #     self.assertNotIn(removed_bid_order_2.uid, lob._orders)
    #     self.assertNotIn(removed_bid_order_2.price, lob._price_levels)
    #
    # def test_adding_buy_order_removes_matching_sell_order(self):
    #     lob = self.lob
    #
    #     # add buy order
    #     order = Order(uid=7, is_bid=True, size=10, price=201.10)
    #
    #     lob.process(order)
    #
    # def check_levels_format(self, levels):
    #     self.assertIsInstance(levels, dict)
    #     for side in ('bids', 'asks'):
    #         self.assertIsInstance(levels[side], list)
    #         for i, price_level in enumerate(levels[side]):
    #             price = price_level.price
    #             last_price = price if i < 1 else levels[side][i - 1].price
    #             if side == 'bids':
    #                 self.assertTrue(price <= last_price)
    #             else:
    #                 self.assertTrue(price >= last_price)
    #
    # def test_querying_levels_works(self):
    #     lob = self.lob
    #     levels = lob.levels()
    #     self.check_levels_format(levels)
    #
    # def test_querying_levels_limit_depth(self):
    #     # Arrange
    #     lob = self.lob
    #
    #     # Act
    #     levels = lob.levels(depth=2)
    #     self.check_levels_format(levels)
    #
    #     # Assert
    #     for side in ('bids', 'asks'):
    #         self.assertEqual(len(levels[side]), 2)
    #
    # def test_querying_levels_by_price(self):
    #     # Arrange
    #     lob = self.lob
    #
    #     expected_result = {
    #         'bids': {
    #             110: 1,
    #             100: 1,
    #             90: 1
    #         },
    #         'asks': {
    #             200: 2,
    #             210: 1
    #         }
    #     }
    #
    #     assert lob.levels_by_price(10) == expected_result
