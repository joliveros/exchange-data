#!/usr/bin/env python

from exchange_data import settings
from exchange_data.emitters import Messenger
from exchange_data.emitters.binance import BinanceUtils
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

cache = RedisCache(redis_client=Redis(host=settings.REDIS_HOST))


class BookTickerEmitter(Messenger, BinanceUtils, BinanceWebSocketApiManager):
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

        self.limit = limit

        self.queued_symbols.update(self.symbols)

        time.sleep(5)

        self.take_symbols(prefix='symbol_emitter', workers=workers)

        alog.info(self.depth_symbols)

        self.create_stream(['bookTicker'], self.depth_symbols)

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
