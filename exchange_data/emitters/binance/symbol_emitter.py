#!/usr/bin/env python

from collections import deque
from datetime import timedelta
from exchange_data import settings
from exchange_data.emitters import Messenger
from exchange_data.emitters.binance import BinanceUtils, ExceededLagException
from exchange_data.utils import DateTimeUtils
from pytimeparse.timeparse import timeparse
from redis import Redis
from redis_cache import RedisCache
from unicorn_binance_websocket_api import BinanceWebSocketApiManager


import click
import json
import signal
import sys
import time
import alog


cache = RedisCache(redis_client=Redis(host=settings.REDIS_HOST))


class SymbolEmitter(Messenger, BinanceUtils, BinanceWebSocketApiManager):
    create_at = None
    depth_symbols = set()
    last_lock_ix = 0
    last_queue_check = None
    stream_id = None

    def __init__(self, limit, workers, **kwargs):
        self.lag_records = deque(maxlen=100)
        if kwargs["futures"]:
            exchange = "binance.com-futures"
        else:
            exchange = "binance.com"

        super().__init__(exchange=exchange, **kwargs)
        BinanceUtils.__init__(self, **kwargs)

        # del kwargs["symbol_filter"]
        del kwargs["symbols"]
        del kwargs["futures"]

        BinanceWebSocketApiManager.__init__(self, exchange=exchange, **kwargs)

        self.limit = limit

        self.update_queued_symbols("symbol")

        self.take_symbols(prefix="symbol_emitter", workers=workers)

        alog.info(self.depth_symbols)

        self.max_lag = 0.5

        self.on("start", self.start_stream)

        self.start_stream()

    def start_stream(self, *args):
        self.last_start = DateTimeUtils.now()
        self.stream_id = self.create_stream(["depth"], self.depth_symbols)
        while True:
            data_str = self.pop_stream_data_from_stream_buffer()

            if data_str:
                self.reset_empty_msg_count()
                data = json.loads(data_str)

                if data:
                    self.handle_data(data, data_str)
            else:
                self.increase_empty_msg_count()
                time.sleep(self.max_lag)

    def handle_data(self, data, data_str):
        if "data" in data:
            if "s" in data["data"]:
                symbol = data["data"]["s"]
                timestamp = DateTimeUtils.parse_db_timestamp(data["data"]["E"])

                lag = DateTimeUtils.now() - timestamp

                self.timing(f"{self.channel_for_symbol(symbol)}_lag", lag)

                self.lag_records.append(lag.total_seconds())

                avg_lag = sum(self.lag_records) / len(self.lag_records)

                if avg_lag > self.max_lag and len(self.lag_records) > 20:
                    alog.info("## acceptable lag has been exceeded ##")
                    self.stream_is_crashing(self.stream_id)

                self.publish(self.channel_for_symbol(symbol), data_str)
        else:
            alog.info(data)

    def channel_for_symbol(self, symbol):
        if self.futures:
            return f"{symbol}_depth_futures"
        else:
            return f"{symbol}_depth"


@click.command()
@click.option("--workers", "-w", default=8, type=int)
@click.option("--limit", "-l", default=0, type=int)
@click.option("--symbols", "-s", nargs=4, type=str)
@click.option("--futures", "-F", is_flag=True)
def main(**kwargs):
    SymbolEmitter(**kwargs)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda *args: exit(0))
    signal.signal(signal.SIGTERM, lambda *args: exit(0))
    main()
