#!/usr/bin/env python
from collections import deque

from cached_property import cached_property
from datetime import timedelta

from redis_collections import Set

from exchange_data import settings
from exchange_data.emitters import Messenger
from exchange_data.emitters.binance import BinanceUtils, ExceededLagException
from exchange_data.utils import DateTimeUtils
from pytimeparse.timeparse import timeparse
from redis import Redis
from redis_cache import RedisCache
from unicorn_binance_websocket_api import BinanceWebSocketApiManager
from unicorn_fy import UnicornFy

import alog
import click
import json
import pickle
import signal
import time
import zlib

cache = RedisCache(redis_client=Redis(host=settings.REDIS_HOST))


class TradeEmitter(Messenger, BinanceUtils, BinanceWebSocketApiManager):
    create_at = None
    depth_symbols = set()
    last_lock_ix = 0
    last_queue_check = None
    stream_id = None

    def __init__(self, limit, workers, **kwargs):
        if kwargs["futures"]:
            exchange = "binance.com-futures"
            self.database_name = "binance_futures"
        else:
            exchange = "binance.com"
            self.database_name = "binance"

        super().__init__(exchange=exchange, **kwargs)
        BinanceUtils.__init__(self, **kwargs)
        # del kwargs["symbol_filter"]

        del kwargs["symbols"]
        del kwargs["futures"]
        BinanceWebSocketApiManager.__init__(self, exchange=exchange, **kwargs)

        self.lag_records = deque(maxlen=100)

        self.limit = limit

        self.update_queued_symbols("trade")

        self.take_symbols(prefix="trade_emitter", workers=workers)

        alog.info(self.depth_symbols)

        self.max_lag = 2

        self.on("start", self.start_stream)

        self.start_stream()

    @cached_property
    def queued_symbols(self):
        return Set(key="trade_queued_symbols", redis=self.redis_client)

    def start_stream(self, *args):
        self._start_stream()
        self.emit("start")

    def _start_stream(self):
        self.last_start = DateTimeUtils.now()
        self.stream_id = self.create_stream(["trade"], self.depth_symbols)

        while True:
            data_str = self.pop_stream_data_from_stream_buffer()
            if data_str:
                data = json.loads(data_str)
                if data:
                    if "data" in data:
                        if "s" in data["data"]:
                            self.is_lag_acceptable(data)
                            obj = pickle.dumps(data["data"])
                            obj = zlib.compress(obj)
                            self.redis_client.lpush(f"{self.database_name}_trades", obj)
            else:
                time.sleep(1 / 10)

    def is_lag_acceptable(self, data):
        timestamp = DateTimeUtils.parse_db_timestamp(data["data"]["E"])
        lag = DateTimeUtils.now() - timestamp

        self.timing("trades_lag", lag)
        self.lag_records.append(lag.total_seconds())

        avg_lag = sum(self.lag_records) / len(self.lag_records)

        # alog.info((len(self.lag_records), avg_lag))

        if avg_lag > self.max_lag and len(self.lag_records) > 20:
            alog.info("## acceptable lag has been exceeded ##")
            self.lag_records.clear()
            self.stream_is_crashing(self.stream_id)

    def channel_for_symbol(self, symbol):
        if self.futures:
            return f"{symbol}_trade_futures"
        else:
            return f"{symbol}_trade"


@click.command()
@click.option("--workers", "-w", default=8, type=int)
@click.option("--limit", "-l", default=0, type=int)
@click.option("--symbols", "-s", nargs=4, type=str)
@click.option("--futures", "-F", is_flag=True)
def main(**kwargs):
    TradeEmitter(**kwargs)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda *args: exit(0))
    signal.signal(signal.SIGTERM, lambda *args: exit(0))
    main()
