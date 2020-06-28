from dateutil import parser
from exchange_data.bitmex_orderbook import InstrumentInfo
from exchange_data.channels import BitmexChannels
from exchange_data.emitters import TimeEmitter
from exchange_data.emitters.bitmex import BinanceOrderBookEmitter
from exchange_data.orderbook.tests.fixtures import orders

import alog
import mock
import pytest


@pytest.fixture()
def instrument_info():
    InstrumentInfo(**{
        'index': 88,
        'symbol': 'XBTUSD',
        'tick_size': 0.5,
        'timestamp': '2018-11-29T03:38:20.000Z'
    })


@pytest.fixture()
@pytest.mark.vcr()
def orderbook_emitter(mocker, orders, tmpdir):
    mocker.patch('exchange_data.bitmex_orderbook.InstrumentInfo'
                 '.get_instrument', return_value=instrument_info)
    mocker.patch(
        'exchange_data.emitters.bitmex._bitmex_orderbook_emitter.Messenger'
    )
    mocker.patch(
        'exchange_data.emitters.bitmex._bitmex_orderbook_emitter'
        '.SignalInterceptor'
    )
    orderbook_emitter = BinanceOrderBookEmitter(BitmexChannels.XBTUSD,
                                                cache_dir=tmpdir)

    for order in orders:
        orderbook_emitter.process_order(order)

    return orderbook_emitter


class TestBitmexOrderBookEmitter(object):
    timestamp = parser.parse('2019-03-06 22:00:03.910369+00:00').timestamp()

    def tick(self):
        one_second = 1000

        if self.timestamp:
            self.timestamp += one_second
            return self.timestamp
        else:
            self.timestamp = TimeEmitter.timestamp()
            return self.timestamp

    def test_generate_orderbook_frame(self, orderbook_emitter):
        frame = orderbook_emitter.generate_frame()

        assert frame.shape == (2, 2, 3)

    @mock.patch('exchange_data.emitters.bitmex.BitmexOrderBookEmitter.write_points')
    def test_save_frame(self, write_points_mock, orderbook_emitter):
        orderbook_emitter.depths = [1]

        orderbook_emitter.save_frame(self.timestamp)

        write_points_mock.called_once()

