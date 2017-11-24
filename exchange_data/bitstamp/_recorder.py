from .. import settings, Recorder
from pysher import Pusher
import alog
import logging

alog.set_level(settings.LOG_LEVEL)


class BitstampRecorder(Pusher, Recorder):
    def __init__(self, symbols):
        self.symbols = [symbol.lower() for symbol in symbols]
        Pusher.__init__(self, settings.BITSTAMP_PUSHER_APP_KEY,
                        log_level=logging.CRITICAL)
        Recorder.__init__(self, self.symbols, database_name='bitstamp')

        self.connection.bind('pusher:connection_established', self.on_connect)
        self.connect()

    def on_connect(self, data):
        for symbol in self.symbols:
            self.subscribe_symbol(symbol)

    def subscribe_symbol(self, symbol):
        self.subscribe('diff_order_book', symbol, ['data'], self.diff_order_book)
        # self.subscribe('order_book', symbol, ['data'], self.order_book)
        # self.subscribe('live_orders', symbol,
        #                ['order_created', 'order_changed', 'order_deleted'],
        #                self.live_orders)
        self.subscribe('live_trades', symbol, ['trade'], self.live_trades)

    def channel_name(self, channel, symbol):
        if symbol == 'btcusd':
            return channel
        return '{}_{}'.format(channel, symbol)

    def subscribe(self, channel, symbol, events, callback):
        channel_name = self.channel_name(channel, symbol)
        channel = super(BitstampRecorder, self).subscribe(channel_name)
        for event in events:
            channel.bind(event, lambda data: callback(symbol, data))
        return channel

    def diff_order_book(self, symbol, data):
        self.save_measurement('diff_order_book', symbol, data)

    def order_book(self, symbol, data):
        self.save_measurement('order_book', symbol, data)

    def live_orders(self, symbol, data):
        self.save_measurement('live_orders', symbol, data)

    def live_trades(self, symbol, data):
        self.save_measurement('live_orders', symbol, data)
