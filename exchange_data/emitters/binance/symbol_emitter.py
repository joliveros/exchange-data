#!/usr/bin/env python
import time
from datetime import timedelta

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
    stream_id = None

    def __init__(
        self,
        limit,
        workers,
        **kwargs
    ):
        super().__init__(exchange="binance.com", **kwargs)
        BinanceWebSocketApiManager.__init__(self, exchange="binance.com", **kwargs)
        self.limit = limit

        self.update_queued_symbols('symbol')

        self.take_symbols(prefix='symbol_emitter', workers=workers)

        alog.info(self.depth_symbols)

        self.max_lag = timeparse('5s')

        self.on('start', self.start_stream)

        self.start_stream()

    def start_stream(self, *args):
        try:
            self._start_stream()
        except ExceededLagException as e:
            self.stop_stream(self.stream_id)
            self.emit('start')

    def _start_stream(self):
        self.last_start = DateTimeUtils.now()
        self.stream_id = self.create_stream(['depth'], self.depth_symbols)
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
                            if self.last_start < DateTimeUtils.now() - \
                                timedelta(seconds=timeparse('1m')):
                                alog.info('## acceptable lag has been exceeded ##')
                                raise ExceededLagException()

                        symbol = data['data']["s"]
                        self.publish(f'{symbol}_depth', data_str)


@click.command()
@click.option('--workers', '-w', default=8, type=int)
@click.option('--limit', '-l', default=0, type=int)
def main(**kwargs):
    SymbolEmitter(**kwargs)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda *args: exit(0))
    signal.signal(signal.SIGTERM, lambda *args: exit(0))
    main()
