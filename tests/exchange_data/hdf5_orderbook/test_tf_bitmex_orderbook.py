import json

import pytest
from mock import patch

from exchange_data import settings
from exchange_data.hdf5_orderbook import \
    Hdf5BitmexLimitOrderBook as OrderBook
from tests.exchange_data.hdf5_orderbook.fixtures import datafile_name


@pytest.fixture('module')
def orderbook():
    _orderbook = OrderBook(symbol='xbtusd',
                           json_file=datafile_name('bitmex'))
    # settings.DB = 'http://jose:jade121415@178.62.16.200:28953/'
    # _orderbook = OrderBook(total_time='1d', symbol='xbtusd', save_json=True)

    return _orderbook


@pytest.fixture('module')
def orderbook_update_msg():
    return {'data': '{"table": "orderBookL2", "action": "update", "data": [{'
                    '"id": '
                    '8799239600, "side": "Sell", "size": 405352}, {"id": '
                    '8799239800, '
                    '"side": "Buy", "size": 162840}], "symbol": "XBTUSD", '
                    '"timestamp": '
                    '"2018-06-09 18:08:54.423294Z"}',
            'symbol': 'XBTUSD',
            'time': 1528567734428}


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


class TestHdf5BitmexLimitOrderBook(object):

    def test_orderbook_message_updates_orderbook(self, orderbook,
                                                 orderbook_update_msg):
        with patch('exchange_data.hdf5_orderbook.Hdf5BitmexLimitOrderBook'
                   '.modify_order') as orderBookL2Mock:

            orderbook.on_message(json.dumps(orderbook_update_msg))

            orderBookL2Mock.assert_called()

    def test_orderbook_message_adds_to_orderbook(self, orderbook,
                                                 orderbook_insert_msg):
        with patch('exchange_data.orderbook.OrderBook'
                   '.process_order') as process_order_mock:
            orderbook.on_message(json.dumps(orderbook_insert_msg))

            process_order_mock.assert_called()

    def test_orderbook_message_deletes_from_orderbook(self, orderbook,
                                                 orderbook_delete_msg):
        with patch('exchange_data.orderbook.OrderBook'
                   '.cancel_order') as cancel_order_mock:

            orderbook.on_message(json.dumps(orderbook_delete_msg))

            cancel_order_mock.assert_called()

    def test_replay(self, orderbook, orderbook_update_msg):
        orderbook.replay()
