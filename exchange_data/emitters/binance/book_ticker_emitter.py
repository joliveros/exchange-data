#!/usr/bin/env python
from collections import deque
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

    def __init__(self, limit, workers, **kwargs):
        super().__init__(exchange="binance.com", **kwargs)
        BinanceUtils.__init__(self, **kwargs)
        del kwargs["futures"]
        del kwargs["symbols"]
        # del kwargs["symbol_filter"]
        BinanceWebSocketApiManager.__init__(self, exchange="binance.com", **kwargs)
        self.lag_records = deque(maxlen=100)
        self.limit = limit

        self.update_queued_symbols("book_ticker")

        self.take_symbols(prefix="book_ticker_emitter", workers=workers)

        alog.info(self.depth_symbols)

        self.max_lag = timeparse("5s")
        self.on("start", self.start_stream)

        self.start_stream()

    @cached_property
    def queued_symbols(self):
        return Set(key="book_ticker_queued_symbols", redis=self.redis_client)

    def start_stream(self, *args):
        self._start_stream()
        self.emit("start")

    def _start_stream(self):
        self.last_start = DateTimeUtils.now()
        self.stream_id = self.create_stream(["ticker"], self.depth_symbols)
        while True:
            data_str = self.pop_stream_data_from_stream_buffer()

            if data_str:
                data = json.loads(data_str)
                if data:
                    if "data" in data:
                        self.handle_data(data)
            else:
                time.sleep(2 / 10)

    def handle_data(self, data):
        data = data["data"]
        if "s" in data:
            symbol = data["s"]
            timestamp = DateTimeUtils.parse_db_timestamp(data["E"])
            lag = DateTimeUtils.now() - timestamp
            self.timing(f"{self.channel_for_symbol(symbol)}_book_ticker_lag", lag)

            self.lag_records.append(lag.total_seconds())

            avg_lag = sum(self.lag_records) / len(self.lag_records)

            if avg_lag > self.max_lag and len(self.lag_records) > 20:
                alog.info("## acceptable lag has been exceeded ##")
                self.stream_is_crashing(self.stream_id)

            self.publish(self.channel_for_symbol(symbol), json.dumps(data))

    def channel_for_symbol(self, symbol):
        if self.futures:
            return f"{symbol}_book_ticker_futures"
        else:
            return f"{symbol}_book_ticker"


@click.command()
@click.option("--limit", "-l", default=0, type=int)
@click.option("--workers", "-w", default=8, type=int)
@click.option("--symbols", "-s", nargs=4, type=str)
@click.option("--futures", "-F", is_flag=True)
def main(**kwargs):
    BookTickerEmitter(**kwargs)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda *args: exit(0))
    signal.signal(signal.SIGTERM, lambda *args: exit(0))
    main()
