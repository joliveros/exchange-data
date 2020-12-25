#!/usr/bin/env python

from exchange_data import Database
from exchange_data.emitters import Messenger, SignalInterceptor
from exchange_data.emitters.binance._orderbook import BinanceOrderBook
from exchange_data.utils import DateTimeUtils

import alog
import click
import sys


class BitmexOrderBookEmitter(
    Messenger,
    Database,
    SignalInterceptor,
    DateTimeUtils,
):
    orderbooks = {}

    def __init__(
        self,
        database_name='binance',
        reset_orderbook: bool = True,
        save_data: bool = True,
        subscriptions_enabled: bool = True,
        **kwargs
    ):
        self.subscriptions_enabled = subscriptions_enabled
        self.should_reset_orderbook = reset_orderbook

        super().__init__(
            database_name=database_name,
            database_batch_size=10,
            **kwargs
        )

        self.save_data = save_data
        self.slices = {}
        self.frame_slice = None
        self.queued_frames = []

        if self.subscriptions_enabled:
            self.on('1m', self.save_measurements)
            self.on('2s', self.temp)
            self.on('depth', self.message)
            self.on('depth_reset', self.depth_reset)

    def temp(self, timestamp):
        symbol = 'ZILBNB'
        if symbol in self.orderbooks:
            alog.info(self.orderbooks[symbol].print(depth=10, trades=False))
            orderbook: BinanceOrderBook = self.orderbooks[symbol]

    def message(self, data):
        if 'data' in data:
            data = data['data']
            symbol = data['s']

            if symbol not in self.orderbooks:
                self.orderbooks[symbol] = BinanceOrderBook(symbol)

            self.orderbooks[symbol].message(data)

    def depth_reset(self, data):
        alog.info('### depth reset ###')
        data = data['data']
        symbol = data['s']
        self.orderbooks[symbol] = BinanceOrderBook(symbol)

        self.orderbooks[symbol].message(data)

    def start(self, channels=[]):
        self.sub([
            '1m',
            '2s',
            'depth',
            'depth_reset'
        ] + channels)

    def exit(self, *args):
        self.stop()
        sys.exit(0)

    def save_measurements(self, timestamp):
        for symbol, book in self.orderbooks.items():
            if symbol == 'ZILBNB':
                alog.info(alog.pformat(book.measurement()))
            self.write_points([book.measurement()], time_precision='ms')


@click.command()
@click.option('--save-data/--no-save-data', default=True)
@click.option('--reset-orderbook/--no-reset-orderbook', default=True)
def main(**kwargs):
    recorder = BitmexOrderBookEmitter(
        subscriptions_enabled=True,
        **kwargs
    )

    recorder.start()

if __name__ == '__main__':
    main()
