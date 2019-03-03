from exchange_data import settings
from exchange_data.bitmex_orderbook import \
    BitmexOrderBook as OrderBook, InstrumentInfo
from exchange_data.channels import BitmexChannels
from tests.exchange_data.fixtures import datafile_name, measurements

import pytest


@pytest.fixture('module')
def orderbook_update_msg():
    return {'action': 'update',
            'data': [{'id': 8799384950,
                      'side': 'Sell',
                      'size': 133981,
                      'symbol': 'XBTUSD'},
                     {'id': 8799388000,
                      'side': 'Buy',
                      'size': 202915,
                      'symbol': 'XBTUSD'}],
            'table': 'orderBookL2'}


@pytest.fixture('module')
def orderbook_insert_msg():
    return {'action': 'insert',
            'data': [{'id': 8799938600,
                      'price': 614,
                      'side': 'Buy',
                      'size': 2,
                      'symbol': 'XBTUSD'}],
            'table': 'orderBookL2'}


@pytest.fixture('module')
def orderbook_delete_msg():
    return {'action': 'delete',
            'data': [{'id': 8799938600, 'side': 'Buy', 'symbol': 'XBTUSD'}],
            'table': 'orderBookL2'}


class TestInstrumentInfo(object):

    @pytest.mark.vcr()
    def test_default_init(self):
        info = InstrumentInfo.get_instrument('XBTUSD')

        assert info.index == 88
        assert info.symbol == 'XBTUSD'
        assert info.tick_size == 0.5


class TestBitmexOrderBook(object):
    @pytest.mark.vcr()
    def test_orderbook_message_updates_orderbook(self,
                                                 orderbook_update_msg,
                                                 mocker):

        orderbook = OrderBook(symbol=BitmexChannels.XBTUSD)

        orderbook.message(orderbook_update_msg)

        assert orderbook.asks.volume == orderbook_update_msg['data'][0]['size']
        assert orderbook.bids.volume == orderbook_update_msg['data'][1]['size']

    # def test_orderbook_message_adds_to_orderbook(self, orderbook_insert_msg,
    #                                              tmpdir, mocker):
    #     mock_process_order = mocker.patch(
    #         'exchange_data.orderbook.OrderBook.process_order'
    #     )
    #     orderbook = OrderBook(total_time='1h', symbol='xbtusd',
    #                           cache_dir=tmpdir, read_from_json=True)
    #     orderbook.message(json.dumps(orderbook_insert_msg))
    #
    #     mock_process_order.assert_called()
    #
    # def test_orderbook_message_deletes_from_orderbook(self,
    #                                                   orderbook_delete_msg,
    #                                                   mocker, tmpdir):
    #     mock_cancel_order = mocker.patch(
    #         'exchange_data.orderbook.OrderBook.cancel_order'
    #     )
    #     orderbook = OrderBook(total_time='1h', symbol='xbtusd',
    #                           cache_dir=tmpdir, read_from_json=True)
    #     orderbook.message(json.dumps(orderbook_delete_msg))
    #
    #     mock_cancel_order.assert_called()
