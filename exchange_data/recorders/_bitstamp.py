import json
import logging

from dateutil.tz import tz

from exchange_data import settings
from exchange_data.utils import DateTimeUtils
from . import Recorder
from pysher import Pusher

import alog

alog.set_level(settings.LOG_LEVEL)


class BitstampRecorder(Pusher, Recorder, DateTimeUtils):
    def __init__(self, symbol):
        self.symbol = symbol
        DateTimeUtils.__init__(self)

        Pusher.__init__(self, settings.BITSTAMP_PUSHER_APP_KEY,
                        log_level=logging.CRITICAL)

        Recorder.__init__(self, self.symbols, database_name='bitstamp')

        self.connection.bind('pusher:connection_established', self.on_connect)

    def start(self):
        self.connection.run()

    def on_connect(self, data):
        self.subscribe_symbol()

    def subscribe_symbol(self):
        symbol = self.symbol
        self.subscribe('diff_order_book', symbol, ['data'],
                       self.diff_order_book)
        self.subscribe('live_trades', symbol, ['trade'], self.live_trades)
        self.subscribe('live_orders', symbol,
                       ['order_created',
                        'order_changed',
                        'order_deleted'], self.live_orders)

    @staticmethod
    def channel_name(channel, symbol):
        if symbol == 'btcusd':
            return channel
        return '{}_{}'.format(channel, symbol)

    def subscribe(self, channel, symbol, events, callback):
        channel_name = self.channel_name(channel, symbol)
        channel = super(BitstampRecorder, self).subscribe(channel_name)
        for event in events:
            channel.bind(event, lambda data: callback(symbol, data))
        return channel

    def parse_data(self, data):
        data = json.loads(data)
        timestamp = int(data['microtimestamp']) / (10 ** 6)
        time = self.parse_timestamp(timestamp, tz=tz.tzutc())

        data['time'] = time
        del data['microtimestamp']

        if 'datetime' in data:
            del data['datetime']

        if 'timestamp' in data:
            del data['timestamp']

        return data

    def diff_order_book(self, symbol, data):
        self.save_measurement(
            'diff_orderbook', symbol, self.parse_data(data)
        )

    def live_orders(self, symbol, data):
        self.save_measurement(
            'live_orders', symbol, self.parse_data(data)
        )

    def live_trades(self, symbol, data):
        self.save_measurement(
            'live_trades', symbol, self.parse_data(data)
        )
