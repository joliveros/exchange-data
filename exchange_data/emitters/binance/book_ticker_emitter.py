#!/usr/bin/env python

from exchange_data import settings
from exchange_data.emitters import Messenger
from exchange_data.emitters.binance import BinanceUtils
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
        **kwargs
    ):
        super().__init__(exchange="binance.com", **kwargs)

        self.create_stream('arr', '!bookTicker')

        while True:
            data_str = self.pop_stream_data_from_stream_buffer()
            data = None

            try:
                data = json.loads(data_str)
            except TypeError as e:
                pass

            if data:
                if 's' in data:
                    symbol = data["s"]
                    self.publish(f'{symbol}_book_ticker', data_str)



@click.command()
def main(**kwargs):
    BookTickerEmitter(**kwargs)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda *args: exit(0))
    signal.signal(signal.SIGTERM, lambda *args: exit(0))
    main()
