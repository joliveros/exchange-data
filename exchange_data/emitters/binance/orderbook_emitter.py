#!/usr/bin/env python
import time

from cached_property import cached_property
from redis_collections import Set
from redlock import RedLockError

from exchange_data import Database
from exchange_data.emitters import Messenger, SignalInterceptor
from exchange_data.emitters.binance import BinanceUtils
from exchange_data.emitters.binance._orderbook import BinanceOrderBook
from exchange_data.orderbook import OrderBookSide
from exchange_data.orderbook.exceptions import PriceDoesNotExistException
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
    last_book_ticker_update = {}

    def __init__(
        self,
        limit,
        workers,
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
            database_batch_size=48,
            **kwargs
        )
        self.limit = limit
        self.save_data = save_data
        self.slices = {}
        self.frame_slice = None
        self.queued_frames = []

        self.update_queued_symbols('orderbook')

        self.take_symbols(prefix='orderbook', workers=workers)

        for symbol in self.depth_symbols:
            self.orderbooks[symbol] = BinanceOrderBook(symbol)

        if self.subscriptions_enabled:
            self.on('15s', self.save_measurements)
            self.on('15s', self.metrics)
            # self.on('2s', self.temp)

            for symbol in self.depth_symbols:
                self.on(f'{symbol}_depth', self.message)
                self.on(f'{symbol}_depth_reset', self.depth_reset)
                self.on(f'{symbol}_book_ticker', self.book_ticker)

    @cached_property
    def queued_symbols(self):
        return Set(key='orderbook_queued_symbols', redis=self.redis_client)

    def temp(self, timestamp):
        symbol = 'ZILBNB'
        if symbol in self.orderbooks:
            alog.info(self.orderbooks[symbol].print(depth=10, trades=False))
            orderbook: BinanceOrderBook = self.orderbooks[symbol]

    def metrics(self, timestamp):
        now = DateTimeUtils.now()
        for symbol in self.last_book_ticker_update.keys():
            secs = (now - self.last_book_ticker_update[symbol]).total_seconds()
            self.gauge('last_book_ticker_update', secs)

            if secs <= 60.0:
                self.incr('last_book_ticker_update_count')

        orderbook: BinanceOrderBook
        for symbol, orderbook in self.orderbooks.items():
            secs = (now - orderbook.last_timestamp).total_seconds()
            self.gauge('last_orderbook_update', secs)

            if secs <= 60.0:
                self.incr('last_orderbook_update_count')

            self.incr('orderbook_count')

    def message(self, data):
        if 'data' in data:
            symbol = data['data']['s']
            data = data['data']
            self.orderbooks[symbol].message(data)

    def book_ticker(self, data):
        best_ask = float(data['a'])
        best_ask_qty = float(data['A'])
        best_bid = float(data['b'])
        best_bid_qty = float(data['B'])
        symbol = data['s']

        book: BinanceOrderBook = self.orderbooks[symbol]

        book.update_price(best_bid, best_bid_qty, OrderBookSide.BID,
                          DateTimeUtils.now())

        book.update_price(best_ask, best_ask_qty, OrderBookSide.ASK,
                          DateTimeUtils.now())

        prices_remove = [price for price in book.bids.price_map.keys() if price
                         > best_bid]

        for price in prices_remove:
            try:
                book.bids.remove_price(price)
            except PriceDoesNotExistException:
                pass
            # alog.info(f'removed {price}')

        prices_remove = [price for price in book.asks.price_map.keys() if price
                         < best_ask]

        for price in prices_remove:
            try:
                book.asks.remove_price(price)
            except PriceDoesNotExistException:
                pass
            # alog.info(f'removed {price}')
        self.last_book_ticker_update[symbol] = DateTimeUtils.now()

    def depth_reset(self, data):
        alog.info('### depth reset ###')
        data = data['data']
        symbol = data['s']
        self.orderbooks[symbol] = BinanceOrderBook(symbol)

        self.orderbooks[symbol].message(data)

    def start(self, channels=[]):
        depth_channels = [f'{symbol}_depth' for symbol in self.depth_symbols]
        depth_reset_channels = [f'{symbol}_depth_reset' for symbol in self.depth_symbols]
        depth_reset_channels += [f'{symbol}_book_ticker' for symbol in
                                self.depth_symbols]
        self.sub([
            '15s',
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
@click.option('--limit', '-l', default=0, type=int)
@click.option('--workers', '-w', default=8, type=int)
def main(**kwargs):
    recorder = BitmexOrderBookEmitter(
        subscriptions_enabled=True,
        **kwargs
    )

    recorder.start()


if __name__ == '__main__':
    main()
