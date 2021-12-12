#!/usr/bin/env python
from datetime import timedelta

from cached_property import cached_property
from redis_collections import Set

from exchange_data import settings
from exchange_data.emitters import Messenger
from exchange_data.emitters.binance import BinanceUtils, ExceededLagException
from exchange_data.emitters.binance.symbol_emitter import SymbolEmitter
from pytimeparse.timeparse import timeparse
from redis import Redis
from redis_cache import RedisCache
from unicorn_binance_websocket_api import BinanceWebSocketApiManager

import alog
import click
import json
import signal
import time

from exchange_data.utils import DateTimeUtils

cache = RedisCache(redis_client=Redis(host=settings.REDIS_HOST))


class BookTickerEmitter(Messenger, BinanceWebSocketApiManager, BinanceUtils):
    create_at = None
    depth_symbols = set()
    last_lock_ix = 0
    last_queue_check = None
    stream_id = None

    def __init__(
        self,
        limit,
        workers,
        **kwargs
    ):
        super().__init__(exchange="binance.com", **kwargs)
        BinanceUtils.__init__(self, **kwargs)
        del kwargs['futures']
        del kwargs['symbol_filter']
        BinanceWebSocketApiManager.__init__(self, exchange="binance.com",
                                            **kwargs)
        self.limit = limit

        self.update_queued_symbols('book_ticker')

        self.take_symbols(prefix='book_ticker_emitter', workers=workers)

        alog.info(self.depth_symbols)

        self.max_lag = timeparse('5s')
        self.on('start', self.start_stream)

        self.start_stream()

    @cached_property
    def queued_symbols(self):
        return Set(key='book_ticker_queued_symbols', redis=self.redis_client)

    def start_stream(self, *args):
        try:
            self._start_stream()
        except ExceededLagException as e:
            self.stop_stream(self.stream_id)
            self.emit('start')

    def _start_stream(self):
        self.last_start = DateTimeUtils.now()
        self.stream_id = self.create_stream(['ticker'], self.depth_symbols)
        while True:
            data_str = self.pop_stream_data_from_stream_buffer()
            data = None

            try:
                data = json.loads(data_str)
            except TypeError as e:
                pass

            if data:
                if 'data' in data:
                    data = data['data']
                    if 's' in data:
                        symbol = data["s"]
                        timestamp = DateTimeUtils.parse_db_timestamp(data['E'])
                        lag = DateTimeUtils.now() - timestamp

                        if lag.total_seconds() > self.max_lag:
                            if self.last_start < DateTimeUtils.now() - \
                               timedelta(seconds=timeparse('1m')):
                                alog.info(
                                    '## acceptable lag has been exceeded ##')
                                raise ExceededLagException()

                        self.publish(self.channel_for_symbol(symbol),
                                     json.dumps(data))

    def channel_for_symbol(self, symbol):
        if self.futures:
            return f'{symbol}_book_ticker_futures'
        else:
            return f'{symbol}_book_ticker'


@click.command()
@click.option('--limit', '-l', default=0, type=int)
@click.option('--workers', '-w', default=8, type=int)
@click.option('--symbol-filter', default=None, type=str)
@click.option('--futures', '-F', is_flag=True)
def main(**kwargs):
    BookTickerEmitter(**kwargs)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda *args: exit(0))
    signal.signal(signal.SIGTERM, lambda *args: exit(0))
    main()
