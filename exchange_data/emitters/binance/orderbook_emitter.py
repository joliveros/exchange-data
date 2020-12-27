#!/usr/bin/env python
from functools import cached_property

from redis_collections import Set

from exchange_data import Database
from exchange_data.emitters import Messenger, SignalInterceptor
from exchange_data.emitters.binance import BinanceUtils
from exchange_data.emitters.binance._orderbook import BinanceOrderBook
from exchange_data.utils import DateTimeUtils

import alog
import click
import sys


class BitmexOrderBookEmitter(
    Messenger,
    BinanceUtils,
    Database,
    SignalInterceptor,
    DateTimeUtils,
):
    orderbooks = {}
    depth_symbols = set()

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
            database_batch_size=1,
            **kwargs
        )

        self.save_data = save_data
        self.slices = {}
        self.frame_slice = None
        self.queued_frames = []

        self.queued_symbols.update(self.symbols)
        self.take_symbols(prefix='orderbook')

        for symbol in self.depth_symbols:
            self.orderbooks[symbol] = BinanceOrderBook(symbol)

        if self.subscriptions_enabled:
            self.on('1m', self.save_measurements)
            # self.on('2s', self.temp)

            for symbol in self.depth_symbols:
                self.on(f'{symbol}_depth', self.message)
                self.on(f'{symbol}_depth_reset', self.depth_reset)

    @cached_property
    def queued_symbols(self):
        return Set(key='orderbook_queued_symbols', redis=self.redis_client)

    def temp(self, timestamp):
        symbol = 'ZILBNB'
        if symbol in self.orderbooks:
            alog.info(self.orderbooks[symbol].print(depth=10, trades=False))
            orderbook: BinanceOrderBook = self.orderbooks[symbol]

    def message(self, data):
        if 'data' in data:
            symbol = data['data']['s']
            data = data['data']
            self.orderbooks[symbol].message(data)

    def depth_reset(self, data):
        alog.info('### depth reset ###')
        data = data['data']
        symbol = data['s']
        self.orderbooks[symbol] = BinanceOrderBook(symbol)

        self.orderbooks[symbol].message(data)

    def start(self, channels=[]):
        depth_channels = [f'{symbol}_depth' for symbol in self.depth_symbols]
        depth_reset_channels = [f'{symbol}_depth_reset' for symbol in self.depth_symbols]
        self.sub([
            '1m',
            '2s',
        ] + channels + depth_channels + depth_reset_channels)

    def exit(self, *args):
        self.stop()
        sys.exit(0)

    def save_measurements(self, timestamp):
        alog.info('### save_measurements ###')
        for symbol, book in self.orderbooks.items():
            if symbol == 'ZILBNB':
                alog.info(book.print(depth=10, trades=False))
            self.write_points([book.measurement()], time_precision='s')


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
