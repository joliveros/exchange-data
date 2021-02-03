#!/usr/bin/env python

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

    def __init__(
        self,
        limit,
        workers,
        **kwargs
    ):
        super().__init__(exchange="binance.com", **kwargs)
        BinanceUtils.__init__(self, **kwargs)

        self.limit = limit

        self.queued_symbols.update(self.symbols)

        time.sleep(5)

        self.take_symbols(prefix='book_ticker_emitter', workers=workers)

        alog.info(self.depth_symbols)

        self.max_lag = timeparse('2s')

        self.start_stream()

    def start_stream(self):
        try:
            self._start_stream()
        except ExceededLagException as e:
            self.stop_manager_with_all_streams()
            self.start_stream()

    def _start_stream(self):
        self.create_stream(['ticker'], self.depth_symbols)
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
                            alog.info('## acceptable lag has been exceeded ##')
                            alog.info(len(self.depth_symbols))
                            raise ExceededLagException()

                        self.publish(f'{symbol}_book_ticker', json.dumps(data))

    @property
    def symbols(self):
        symbols = json.loads(SymbolEmitter._symbols())
        if len(symbols) > 0:
            return symbols
        else:
            raise Exception('not enough symbols')

    @staticmethod
    @cache.cache(ttl=timeparse('1h'))
    def _symbols():
        symbols = BinanceUtils().get_symbols()
        return json.dumps(symbols)


@click.command()
@click.option('--limit', '-l', default=0, type=int)
@click.option('--workers', '-w', default=8, type=int)
def main(**kwargs):
    BookTickerEmitter(**kwargs)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda *args: exit(0))
    signal.signal(signal.SIGTERM, lambda *args: exit(0))
    main()
