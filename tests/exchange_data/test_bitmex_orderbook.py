import json

import alog
import pytest
from mock import patch

from exchange_data import settings
from exchange_data.bitmex_orderbook import \
    BitmexOrderBook as OrderBook
from tests.exchange_data.fixtures import datafile_name, measurements

settings.DB = 'http://jose:jade121415@178.62.16.200:28953/'


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


class TestBitmexOrderBook(object):

    def test_orderbook_message_updates_orderbook(self,
                                                 orderbook_update_msg,
                                                 mocker, tmpdir):
        mocked_orderbook_l2 = mocker.patch(
            'exchange_data.orderbook.OrderBook.modify_order'
        )

        orderbook = OrderBook(total_time='1h', symbol='xbtusd',
                              cache_dir=tmpdir, read_from_json=True)

        orderbook.on_message(json.dumps(orderbook_update_msg))

        mocked_orderbook_l2.assert_called()

    def test_orderbook_message_adds_to_orderbook(self, orderbook_insert_msg,
                                                 tmpdir, mocker):
        mock_process_order = mocker.patch(
            'exchange_data.orderbook.OrderBook.process_order'
        )
        orderbook = OrderBook(total_time='1h', symbol='xbtusd',
                              cache_dir=tmpdir, read_from_json=True)
        orderbook.on_message(json.dumps(orderbook_insert_msg))

        mock_process_order.assert_called()

    def test_orderbook_message_deletes_from_orderbook(self,
                                                      orderbook_delete_msg,
                                                      mocker, tmpdir):
        mock_cancel_order = mocker.patch(
            'exchange_data.orderbook.OrderBook.cancel_order'
        )
        orderbook = OrderBook(total_time='1h', symbol='xbtusd',
                              cache_dir=tmpdir, read_from_json=True)
        orderbook.on_message(json.dumps(orderbook_delete_msg))

        mock_cancel_order.assert_called()

    def test_fetch_and_save(self, mocker, tmpdir):
        orderbook = OrderBook(total_time='1h', symbol='xbtusd',
                              cache_dir=tmpdir, read_from_json=True)

