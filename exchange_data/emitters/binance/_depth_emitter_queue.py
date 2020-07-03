#!/usr/bin/env python

from binance.client import Client
from binance.depthcache import DepthCacheManager
from binance.exceptions import BinanceAPIException
from cached_property import cached_property
from datetime import timedelta, datetime

from redlock import RedLock, RedLockError

from exchange_data import settings
from exchange_data.emitters import Messenger
from exchange_data.utils import DateTimeUtils
from pytimeparse.timeparse import timeparse
from redis_collections import List, Dict, Set

import alog
import click
import json
import numpy as np
import os
import signal
import time


class DepthEmitterQueue(Messenger):
    def __init__(self, timeout_interval, **kwargs):
        timeout_seconds = timeparse(timeout_interval)
        self.timeout_interval = timedelta(seconds=timeout_seconds)

        super().__init__(**kwargs)

        self.symbols = Dict(key='symbols', redis=self.redis_client)
        self.symbols_queue = Set(key='symbols_queue', redis=self.redis_client)

        for symbol in self.get_symbols():
            self.symbols[symbol] = None

        while True:
            for symbol, timestamp in self.symbols.items():
                if timestamp is not None:
                    timestamp = DateTimeUtils.parse_datetime_str(str(timestamp))

                last_update = DateTimeUtils.now() - self.timeout_interval

                alog.info(timestamp is None or timestamp < last_update)

                if timestamp is None or timestamp < last_update:
                    self.symbols[symbol] = DateTimeUtils.now()
                    self.symbols_queue.add(symbol)

            alog.info(f'### queue length {len(self.symbols_queue)}')

            time.sleep(timeout_seconds)

    @cached_property
    def client(self):
        return Client()

    def get_symbols(self):
        try:
            return self._get_symbols()
        except BinanceAPIException as e:
            time.sleep(30)
            return self.get_symbols()

    def _get_symbols(self):
        exchange_info = self.client.get_exchange_info()

        symbols = [symbol for symbol in exchange_info['symbols']
                   if symbol['status'] == 'TRADING']

        symbol_names = [symbol['symbol'] for symbol in symbols if symbol[
            'symbol']]

        return symbol_names


@click.command()
@click.option('--delay', '-d', type=str, default='4s')
@click.option('--timeout-interval', '-t', type=str, default='30s')
@click.option('--num-locks', '-n', type=int, default=4)
def main(**kwargs):
    emitter = DepthEmitterQueue(**kwargs)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda: exit(0))
    signal.signal(signal.SIGTERM, lambda: exit(0))
    main()
