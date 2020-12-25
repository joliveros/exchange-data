#!/usr/bin/env python
import time

from exchange_data import settings
from exchange_data.emitters import Messenger
from exchange_data.emitters.binance import BinanceUtils
from pytimeparse.timeparse import timeparse
from redis import Redis
from redis_cache import RedisCache
from redis_collections import Set
from redlock import RedLock
from unicorn_binance_websocket_api import BinanceWebSocketApiManager
import alog
import click
import json
import signal


cache = RedisCache(redis_client=Redis(host=settings.REDIS_HOST))


class SymbolEmitter(Messenger, BinanceUtils, BinanceWebSocketApiManager):
    create_at = None
    depth_symbols = set()
    last_lock_ix = 0
    last_queue_check = None

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(exchange="binance.com", **kwargs)

        self.queued_symbols = Set(key='queued_symbols', redis=self.redis_client)

        self.queued_symbols.update(self.symbols)

        self.take_symbols()

        self.create_stream(['depth'], self.depth_symbols)

        while True:
            data = self.pop_stream_data_from_stream_buffer()

            if data:
                self.publish('depth', data)

    @property
    def symbols(self):
        return json.loads(SymbolEmitter._symbols())

    @staticmethod
    @cache.cache(ttl=timeparse('1h'))
    def _symbols():
        symbols = BinanceUtils().get_symbols()
        return json.dumps(symbols)

    def take_symbols(self):
        while len(self.queued_symbols) > 0:
            queued_symbols = len(self.queued_symbols)
            take_count = 10

            if queued_symbols < take_count:
                take_count = queued_symbols

            for i in range(0, take_count):
                try:
                    symbol = self.queued_symbols.pop()
                    self.depth_symbols.add(symbol)
                except KeyError as e:
                    break

            time.sleep(2)

        alog.info(len(self.depth_symbols))


@click.command()
def main(**kwargs):
    SymbolEmitter(**kwargs)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda *args: exit(0))
    signal.signal(signal.SIGTERM, lambda *args: exit(0))
    main()
