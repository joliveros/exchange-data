import alog

from exchange_data.bitmex_orderbook import \
    BitmexOrderBook as OrderBook, InstrumentInfo
from exchange_data.channels import BitmexChannels

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


@pytest.fixture()
def buy_trade():
    return {
        'action': 'insert',
        'data': [{'foreignNotional': 1,
                  'grossValue': 25961,
                  'homeNotional': 0.00025961,
                  'price': 3852,
                  'side': 'Buy',
                  'size': 1,
                  'symbol': 'XBTUSD',
                  'tickDirection': 'PlusTick',
                  'timestamp': '2019-03-06T22:56:43.268Z',
                  'trdMatchID': '6363afe3-ac45-3047-0ce9-20b3e22e48a0'}],
        'table': 'trade'}


@pytest.fixture()
def sell_trade():
    return {
        'action': 'insert',
        'data': [{'foreignNotional': 600,
                  'grossValue': 15578400,
                  'homeNotional': 0.155784,
                  'price': 3851.5,
                  'side': 'Sell',
                  'size': 600,
                  'symbol': 'XBTUSD',
                  'tickDirection': 'MinusTick',
                  'timestamp': '2019-03-06T22:56:41.839Z',
                  'trdMatchID': '315916d3-0f5c-3c23-2ff7-afaaf831a785'}],
        'table': 'trade'}


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

    def test_orderbook_message_adds_to_orderbook(self, orderbook_insert_msg, mocker):
        mock_process_order = mocker.patch(
            'exchange_data.orderbook.OrderBook.process_order'
        )
        orderbook = OrderBook(symbol=BitmexChannels.XBTUSD)
        orderbook.message(orderbook_insert_msg)

        mock_process_order.assert_called()

    def test_orderbook_message_deletes_from_orderbook(self,
                                                      orderbook_delete_msg,
                                                      mocker):
        mock_cancel_order = mocker.patch(
            'exchange_data.bitmex_orderbook.BitmexOrderBook.cancel_order'
        )
        orderbook = OrderBook(symbol=BitmexChannels.XBTUSD)
        orderbook.message(orderbook_delete_msg)

        mock_cancel_order.assert_called()

    def test_buy_trade(self, buy_trade):
        orderbook = OrderBook(symbol=BitmexChannels.XBTUSD)
        orderbook.message(buy_trade)
