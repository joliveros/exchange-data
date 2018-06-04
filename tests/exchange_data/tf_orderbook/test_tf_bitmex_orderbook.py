import pytest
from mock import patch

from exchange_data.tf_orderbook._tf_bitmex_orderbook import \
    TFBitmexLimitOrderBook as OrderBook
from tests.exchange_data.tf_orderbook.fixtures import datafile_name


@pytest.fixture('module')
def orderbook():
    _orderbook = OrderBook(symbol='xbtusd',
                           json_file=datafile_name('bitmex'))

    # _orderbook = OrderBook(total_time='1h', symbol='xbtusd', save_json=True)

    return _orderbook


@pytest.fixture('module')
def orderbook_update_msg():
    return {'time': 1524614754656,
            'data': '{"table": "orderBookL2", '
                    '"action": "update", '
                    '"data": [{"id": '
                    '8799033450, "side": '
                    '"Sell", "size": 8848}, '
                    '{"id": 8799033600, "side": '
                    '"Sell", "size": 9595}, '
                    '{"id": 8799036800, "side": '
                    '"Buy", "size": 1399}, '
                    '{"id": 8799036950, "side": '
                    '"Buy", "size": 143591}], '
                    '"symbol": "XBTUSD", '
                    '"timestamp": "2018-04-25 '
                    '00:05:54.651939Z"}',
            'symbol': 'XBTUSD'}


@pytest.fixture('module')
def orderbook_insert_msg():
    return {'time': 1524614720860,
            'data': '{"table": "orderBookL2", "action": '
                    '"insert", "data": [{"id": 8799034650, '
                    '"side": "Sell", "size": 8688, "price": 9653.5}], '
                    '"symbol": "XBTUSD", "timestamp": "2018-04-25 '
                    '00:05:20.856051Z"}',
            'symbol': 'XBTUSD'}


@pytest.fixture('module')
def orderbook_delete_msg():
    return {'time': 1524614684960,
            'data': '{"table": "orderBookL2", '
                    '"action": "delete", "data": [{'
                    '"id": 8799030450, "side": '
                    '"Sell"}, {"id": 8799030900, '
                    '"side": "Sell"}], "symbol": '
                    '"XBTUSD", "timestamp": '
                    '"2018-04-25 00:04:44.955633Z"}',
            'symbol': 'XBTUSD'}


# class TestTFBitmexLimitOrderBook(object):
#
#     def test_orderbook_message_updates_orderbook(self, orderbook,
#                                                  orderbook_update_msg):
#         with patch('exchange_data.tf_orderbook.TFBitmexLimitOrderBook'
#                    '.order_book_l2') as orderBookL2Mock:
#             orderbook.on_message(orderbook_update_msg)
#
#             orderBookL2Mock.assert_called_once()
#
#     def test_orderbook_message_adds_to_orderbook(self, orderbook,
#                                                  orderbook_insert_msg):
#         with patch('exchange_data.orderbook.OrderBook'
#                    '.process_order') as insert_mock:
#             orderbook.on_message(orderbook_insert_msg)
#
#             insert_mock.assert_called()
#
#     def test_orderbook_message_deletes_from_orderbook(self, orderbook,
#                                                  orderbook_delete_msg):
#         with patch('exchange_data.orderbook.OrderBook'
#                    '.process_order') as delete_mock:
#             orderbook.on_message(orderbook_delete_msg)
#
#             delete_mock.assert_called()

    # def test_message_trade(self, orderbook):
    #     insert = {'time': 1524614709875,
    #               'data': '{"table": "trade", "action": "insert", "data": [{'
    #                       '"timestamp": "2018-04-25T00:05:06.858Z", "side": '
    #                       '"Sell", "size": 200, "price": 9665, '
    #                       '"tickDirection": "ZeroMinusTick", "trdMatchID": '
    #                       '"02b11565-2b07-b2ca-0e30-077c58da3641", '
    #                       '"grossValue": 2069400, "homeNotional": 0.020694, '
    #                       '"foreignNotional": 200}], "symbol": "XBTUSD", '
    #                       '"timestamp": "2018-04-25 00:05:09.870297Z"}',
    #               'symbol': 'XBTUSD'}
    #
    #     with patch('exchange_data.limit_orderbook.LimitOrderBook'
    #                '.trade') as trade_mock:
    #         orderbook.on_message(insert)
    #
    #         trade_mock.assert_called()
