import alog

from exchange_data.emitters import TimeEmitter
from exchange_data.emitters.bitmex import BitmexChannels
from exchange_data.emitters.bitmex._bitmex_orderbook_emitter import \
    BitmexOrderBookEmitter
from exchange_data.orderbook import BuyOrder, OrderList
from tests.exchange_data.orderbook.fixtures import orders


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

    def test_generate_orderbook_frame(self, orders, mocker):
        mocker.patch(
            'exchange_data.emitters.bitmex._bitmex_orderbook_emitter.Messenger'
        )
        orderbook_emitter = BitmexOrderBookEmitter(BitmexChannels.XBTUSD)

        for order in orders:
            orderbook_emitter.process_order(order)

        frame = orderbook_emitter.generate_frame()

        assert frame.shape == (2, 6)

    def test_update_dataset_once(self, orders, mocker):
        mocker.patch(
            'exchange_data.emitters.bitmex._bitmex_orderbook_emitter.Messenger'
        )
        orderbook_emitter = BitmexOrderBookEmitter(BitmexChannels.XBTUSD)

        for order in orders:
            orderbook_emitter.process_order(order)

        timestamp = TimeEmitter.timestamp()

        dataset = orderbook_emitter.update_dataset(timestamp)

        assert list(dataset.dims) == ['price', 'time', 'volume']

    def test_update_dataset_multiple_times(self, orders, mocker):
        mocker.patch(
            'exchange_data.emitters.messenger.Redis'
        )
        orderbook_emitter = BitmexOrderBookEmitter(BitmexChannels.XBTUSD)

        for order in orders:
            orderbook_emitter.process_order(order)

        timestamp = TimeEmitter.timestamp()

        orderbook_emitter.update_dataset(timestamp)
        one_second = 1000
        next_timestamp = timestamp + one_second
        orderbook_emitter.update_dataset(next_timestamp)

        orderbook_emitter.process_order(BuyOrder(price=91.00, quantity=5))

        orderbook_emitter.update_dataset(next_timestamp + one_second)

    def test_update_dataset_multiple_times_and_delete_level(
            self,
            orders,
            mocker
    ):
        mocker.patch(
            'exchange_data.emitters.messenger.Redis'
        )
        orderbook = BitmexOrderBookEmitter(BitmexChannels.XBTUSD)

        for order in orders:
            orderbook.process_order(order)

        orderbook.update_dataset(self.tick())

        orderbook.update_dataset(self.tick())

        orderbook.process_order(BuyOrder(price=91.00, quantity=5))

        orderbook.update_dataset(self.tick())

        min_price = orderbook.bids.min_price()
        min_level: OrderList = orderbook.bids.price_map[min_price]

        orderbook.cancel_order(min_level.head_order.uid)

        orderbook.update_dataset(self.tick())

    def test_trim_dataset(
            self,
            orders,
            mocker
    ):
        mocker.patch(
            'exchange_data.emitters.messenger.Redis'
        )
        orderbook = BitmexOrderBookEmitter(
            BitmexChannels.XBTUSD,
            max_dataset_length='2s'
        )

        for order in orders:
            orderbook.process_order(order)

        orderbook.update_dataset(self.tick())

        orderbook.update_dataset(self.tick())

        orderbook.process_order(BuyOrder(price=91.00, quantity=5))

        orderbook.update_dataset(self.tick())

        orderbook.update_dataset(self.tick())

        orderbook.update_dataset(self.tick())

        assert orderbook.max_dataset_length == len(orderbook.dataset.time)
