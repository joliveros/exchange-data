#!/usr/bin/env python
import time

from exchange_data import settings
from exchange_data.emitters import Messenger
from exchange_data.emitters.binance import BinanceUtils, ExceededLagException
from pytimeparse.timeparse import timeparse
from redis import Redis
from redis_cache import RedisCache
from unicorn_binance_websocket_api import BinanceWebSocketApiManager
import alog
import click
import json
import signal

from exchange_data.utils import DateTimeUtils

cache = RedisCache(redis_client=Redis(host=settings.REDIS_HOST))


class SymbolEmitter(Messenger, BinanceUtils, BinanceWebSocketApiManager):
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

        self.max_lag = timeparse('2s')

        self.start_stream()

    def start_stream(self):
        try:
            self._start_stream()
        except ExceededLagException as e:
            self.stop_manager_with_all_streams()
            self.start_stream()

    def _start_stream(self):
        self.create_stream(['depth'], self.depth_symbols)
        while True:
            data_str = self.pop_stream_data_from_stream_buffer()
            data = None
            try:
                data = json.loads(data_str)
            except TypeError as e:
                pass

            if data:
                if 'data' in data:
                    if 's' in data['data']:
                        timestamp = DateTimeUtils.parse_db_timestamp(
                            data['data']['E']
                        )

                        lag = DateTimeUtils.now() - timestamp

                        if lag.total_seconds() > self.max_lag:
                            alog.info('## acceptable lag has been exceeded ##')
                            raise ExceededLagException()

                        symbol = data['data']["s"]
                        self.publish(f'{symbol}_depth', data_str)

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
@click.option('--workers', '-w', default=8, type=int)
@click.option('--limit', '-l', default=0, type=int)
def main(**kwargs):
    SymbolEmitter(**kwargs)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda *args: exit(0))
    signal.signal(signal.SIGTERM, lambda *args: exit(0))
    main()
