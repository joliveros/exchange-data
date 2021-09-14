#!/usr/bin/env python
from cached_property import cached_property
from exchange_data import Database
from exchange_data.emitters import Messenger, SignalInterceptor
from exchange_data.emitters.binance import BinanceUtils
from exchange_data.emitters.binance._orderbook import BinanceOrderBook
from exchange_data.orderbook import OrderBookSide
from exchange_data.orderbook.exceptions import PriceDoesNotExistException
from exchange_data.utils import DateTimeUtils
from redis_collections import Set

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
        max_depth: int,
        symbol: str = None,
        save_data: bool = True,
        subscriptions_enabled: bool = True,
        **kwargs
    ):
        self.subscriptions_enabled = subscriptions_enabled

        stats_prefix = None

        if kwargs['futures']:
            database_name = 'binance_futures'
            stats_prefix = 'futures'
        else:
            database_name = 'binance'

        super().__init__(
            retry_on_timeout=True,
            socket_keepalive=True,
            database_name=database_name,
            database_batch_size=100,
            stats_prefix=stats_prefix,
            **kwargs
        )

        self.symbol = symbol
        self.limit = limit
        self.save_data = save_data
        self.slices = {}
        self.frame_slice = None
        self.queued_frames = []
        self.measurements = []
        self.m_measurements = []

        if symbol:
            self.depth_symbols.add(self.symbol)
            self.orderbooks[symbol] = \
                BinanceOrderBook(symbol, max_depth=max_depth)
        else:
            self.update_queued_symbols('orderbook')
            self.take_symbols(prefix='orderbook', workers=workers)
            for symbol in self.depth_symbols:
                self.orderbooks[symbol] = \
                    BinanceOrderBook(symbol, max_depth=max_depth)

        if self.subscriptions_enabled:
            self.on('1m', self.save_measurements_1m)
            self.on('10s', self.save_measurements)
            self.on('10s', self.metrics)

            if self.symbol:
                self.on('2s', self.temp)

            for symbol in self.depth_symbols:
                alog.info(self.channel_suffix(f'{symbol}_depth'))

                self.on(self.channel_suffix(f'{symbol}_depth'), self.message)
                # self.on(self.channel_suffix(f'{symbol}_depth_reset'), self.depth_reset)
                self.on(self.channel_suffix(f'{symbol}_book_ticker'), self.book_ticker)

    @cached_property
    def queued_symbols(self):
        return Set(key='orderbook_queued_symbols', redis=self.redis_client)

    def temp(self, timestamp):
            alog.info(self.orderbooks[self.symbol]
                      .print(depth=84, trades=False))

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
        data = data['data']
        symbol = data['s']
        self.orderbooks[symbol] = BinanceOrderBook(symbol)

        self.orderbooks[symbol].message(data)

    def spot_channels(self):
        depth_channels = [f'{symbol}_depth' for symbol in self.depth_symbols]
        depth_reset_channels = [
            f'{symbol}_depth_reset' for symbol in self.depth_symbols]
        depth_reset_channels += [f'{symbol}_book_ticker' for symbol in
                                self.depth_symbols]
        channels = depth_channels + depth_reset_channels
        return channels

    def futures_channels(self):
        depth_channels = [
            f'{symbol}_depth_futures' for symbol in self.depth_symbols]
        depth_reset_channels = [
            f'{symbol}_depth_reset_futures' for symbol in self.depth_symbols]
        depth_reset_channels += [
            f'{symbol}_book_ticker_futures' for symbol in self.depth_symbols]
        return depth_channels + depth_reset_channels

    def start(self, channels=[]):
        if self.futures:
            channels = channels + self.futures_channels()
        else:
            channels = channels + self.spot_channels()

        self.sub([
            '1m',
            '10s',
            '2s',
        ] + channels)

    def exit(self, *args):
        self.stop()
        sys.exit(0)

    def save_measurements_1m(self, timestamp):
        try:
            self.save_measurements(timestamp,
                                   measurements=self.m_measurements,
                                   database=f'{self.database_name}_1m')
        except Exception:
            self.save_measurements_1m(timestamp)

    def save_measurements(self, timestamp, measurements=None, **kwargs):
        if measurements is None:
            measurements = self.measurements

        for symbol, book in self.orderbooks.items():
            measurements.append(book.measurement())

        try:
            self.write_points(measurements, time_precision='s', **kwargs)
            measurements = []
        except Exception:
            self.save_measurements(timestamp, **kwargs)


@click.command()
@click.option('--futures', '-F', is_flag=True)
@click.option('--limit', '-l', default=0, type=int)
@click.option('--log-requests', is_flag=True)
@click.option('--max-depth', '-m', default=8, type=int)
@click.option('--save-data/--no-save-data', default=True)
@click.option('--symbol', default=None, type=str)
@click.option('--symbol-filter', default=None, type=str)
@click.option('--workers', '-w', default=8, type=int)
def main(**kwargs):
    recorder = BitmexOrderBookEmitter(
        subscriptions_enabled=True,
        **kwargs
    )

    recorder.start()


if __name__ == '__main__':
    main()
