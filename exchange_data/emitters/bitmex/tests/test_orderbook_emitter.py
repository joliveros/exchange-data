from unittest.mock import MagicMock

import alog
import pytest

from exchange_data.bitmex_orderbook import InstrumentInfo
from exchange_data.emitters import TimeEmitter
from exchange_data.channels import BitmexChannels
from exchange_data.emitters.bitmex import BitmexOrderBookEmitter
from exchange_data.orderbook import BuyOrder, OrderList
from tests.exchange_data.orderbook.fixtures import orders


@pytest.fixture()
def instrument_info():
    InstrumentInfo(**{
        'index': 88,
        'symbol': 'XBTUSD',
        'tick_size': 0.5,
        'timestamp': '2018-11-29T03:38:20.000Z'
    })


@pytest.fixture()
def orderbook_emitter(mocker, orders, tmpdir):
    mocker.patch('exchange_data.bitmex_orderbook.InstrumentInfo'
                 '.get_instrument', return_value=instrument_info)
    mocker.patch(
        'exchange_data.emitters.bitmex._bitmex_orderbook_emitter.Messenger'
    )
    mocker.patch(
        'exchange_data.emitters.bitmex.BitmexOrderBookEmitter.publish'
    )
    orderbook_emitter = BitmexOrderBookEmitter(BitmexChannels.XBTUSD,
                                               cache_dir=tmpdir)

    for order in orders:
        orderbook_emitter.process_order(order)

    return orderbook_emitter


class TestBitmexOrderBookEmitter(object):
    timestamp = None

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

    def test_update_dataset_once(self, orderbook_emitter):
        timestamp = TimeEmitter.timestamp()

        dataset = orderbook_emitter.update_dataset(timestamp)

        assert list(dataset.dims) == ['frame', 'levels', 'side', 'time']

    def test_update_dataset_multiple_times(self, orderbook_emitter):
        timestamp = TimeEmitter.timestamp()

        orderbook_emitter.update_dataset(timestamp)
        one_second = 1000
        next_timestamp = timestamp + one_second
        orderbook_emitter.update_dataset(next_timestamp)

        orderbook_emitter.process_order(BuyOrder(price=91.00, quantity=5))

        orderbook_emitter.update_dataset(next_timestamp + one_second)

    def test_update_dataset_multiple_times_and_delete_level(self,
                                                            orderbook_emitter):
        orderbook_emitter.update_dataset(self.tick())
        orderbook_emitter.update_dataset(self.tick())
        orderbook_emitter.process_order(BuyOrder(price=91.00, quantity=5))
        orderbook_emitter.update_dataset(self.tick())

        min_price = orderbook_emitter.bids.min_price()
        min_level: OrderList = orderbook_emitter.bids.price_map[min_price]

        orderbook_emitter.cancel_order(min_level.head_order.uid)

        orderbook_emitter.update_dataset(self.tick())

    def test_emit_last_frame(self, orderbook_emitter: BitmexOrderBookEmitter):
        orderbook_emitter.update_dataset(self.tick())
        orderbook_emitter.update_dataset(self.tick())
        orderbook_emitter.process_order(BuyOrder(price=91.00, quantity=5))
        orderbook_emitter.update_dataset(self.tick())
        orderbook_emitter.update_dataset(self.tick())
        orderbook_emitter.update_dataset(self.tick())

        assert orderbook_emitter.publish.call_count == 5

    def test_open_and_append(self, mocker, orders, tmpdir):
        mocker.patch('exchange_data.bitmex_orderbook.InstrumentInfo'
                     '.get_instrument', return_value=instrument_info)
        mocker.patch(
            'exchange_data.emitters.bitmex._bitmex_orderbook_emitter.Messenger'
        )
        mocker.patch(
            'exchange_data.emitters.bitmex.BitmexOrderBookEmitter.publish'
        )

        args = {
            'symbol': BitmexChannels.XBTUSD,
            'cache_dir': tmpdir,
            'save_interval': '1s'
        }
        orderbook_emitter = BitmexOrderBookEmitter(**args)
        orderbook_emitter._pubsub = MagicMock()

        for order in orders:
            orderbook_emitter.process_order(order)

        orderbook_emitter.update_dataset(self.tick())

        orderbook_emitter.update_dataset(self.tick())

        orderbook_emitter.process_order(BuyOrder(price=91.00, quantity=5))
        orderbook_emitter.update_dataset(self.tick())
        orderbook_emitter.update_dataset(self.tick())
        orderbook_emitter.update_dataset(self.tick())

        orderbook_emitter.stop()

        orderbook_emitter = BitmexOrderBookEmitter(**args)
        orderbook_emitter._pubsub = MagicMock()

        orderbook_emitter.update_dataset(self.tick())

        orderbook_emitter.stop()

