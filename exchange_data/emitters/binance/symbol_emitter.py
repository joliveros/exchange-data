#!/usr/bin/env python
import json

import alog
from pytimeparse.timeparse import timeparse
from redlock import RedLock

from exchange_data import settings
from exchange_data.emitters import Messenger
from exchange_data.emitters.binance import BinanceUtils
from redis import Redis
from redis_cache import RedisCache
from unicorn_binance_websocket_api import BinanceWebSocketApiManager

import click
import signal


cache = RedisCache(redis_client=Redis(host=settings.REDIS_HOST))


class SymbolEmitter(Messenger, BinanceUtils, BinanceWebSocketApiManager):
    create_at = None
    last_queue_check = None

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(exchange="binance.com", **kwargs)

        alog.info(alog.pformat(self.symbols))

        raise Exception()

        self.create_stream(['depth'], self.symbols)

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

    def task_index_lock(self):
        lock_name = f'task_index_lock'

        lock = RedLock(lock_name, [dict(
            host=settings.REDIS_HOST,
            db=0
        )], retry_delay=300, retry_times=3, ttl=timeparse('30s') * 1000)

        alog.info(lock_name)

        return lock




@click.command()
def main(**kwargs):
    emitter = SymbolEmitter(**kwargs)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda *args: exit(0))
    signal.signal(signal.SIGTERM, lambda *args: exit(0))
    main()
