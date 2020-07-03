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


class DepthEmitter(Messenger):
    def __init__(self, delay, num_locks=2, **kwargs):
        super().__init__(**kwargs)

        delay = int(timeparse(delay))
        self.num_locks = num_locks
        self.last_lock_id = 0
        self.lock = None
        self.symbols = Dict(key='symbols', redis=self.redis_client)
        self.symbols_queue = Set(key='symbols_queue', redis=self.redis_client)
        self.caches = {}
        self.cache_queue = []

        alog.info('### initialized ###')

        while True:
            alog.info('### check queues ###')
            self.update_symbol_timestamps()

            if len(self.symbols_queue) > 0:
                symbol = self.symbols_queue.pop()
                self.symbols[symbol] = DateTimeUtils.now()
                self.add_cache(symbol)

                cache_symbols = list(self.caches.keys())

                for sym in cache_symbols:
                    if sym in self.symbols_queue and sym != symbol:
                        alog.info(f'### removing {sym} ###')
                        self.caches[sym].close(close_socket=True)
                        del self.caches[sym]

            time.sleep(1)

    def update_symbol_timestamps(self):
        for symbol, cache in self.caches.items():
            self.symbols[symbol] = cache.last_update_timestamp

    @cached_property
    def client(self):
        return Client()

    def add_cache_lock(self):
        self.last_lock_id = self.last_lock_id + 1

        if self.last_lock_id > self.num_locks - 1:
            self.last_lock_id = 0
        lock_name = f'add_cache_lock_{self.last_lock_id}'

        alog.info(lock_name)

        return RedLock(lock_name, [dict(
            host=settings.REDIS_HOST,
            db=0
        )], retry_delay=200, retry_times=3)

    def add_cache(self, symbol):
        try:
            self._add_cache(symbol)

        except BinanceAPIException as e:
            alog.info(alog.pformat(vars(e)))
            if e.status_code == 418:
                time.sleep(30)
            self.symbols[symbol] = None

        except RedLockError as e:
            self.add_cache(symbol)

    def _add_cache(self, symbol):
        with self.add_cache_lock():
            alog.info(f'### add cache {symbol}###')

            if symbol in self.caches.keys():
                self.caches[symbol].close(close_socket=True)

            self.caches[symbol] = DepthCacheManager(
                self.client,
                callback=self.message,
                limit=5000,
                refresh_interval=timeparse('1h'),
                symbol=symbol,
            )

            alog.info(f'### end add cache {symbol}###')

    def get_symbols(self):
        exchange_info = self.client.get_exchange_info()

        symbols = [symbol for symbol in exchange_info['symbols']
                   if symbol['status'] == 'TRADING']

        symbol_names = [symbol['symbol'] for symbol in symbols if symbol[
            'symbol']]

        return symbol_names

    def exit(self):
        os.kill(os.getpid(), signal.SIGKILL)

    def message(self, depthCache):
        if depthCache is None:
            self.exit()

        symbol = depthCache.symbol
        asks = np.expand_dims(np.asarray(depthCache.get_asks()), axis=0)
        bids = np.expand_dims(np.asarray(depthCache.get_bids()), axis=0)
        ask_levels = asks.shape[1]
        bid_levels = bids.shape[1]

        if ask_levels > bid_levels:
            bids = np.resize(bids, asks.shape)

        elif bid_levels > ask_levels:
            asks = np.resize(asks, bids.shape)

        depth = np.concatenate((asks, bids))

        # alog.info(depth)
        # alog.info(symbol)

        msg = dict(
            symbol=symbol,
            depth=depth.tolist()
        )

        self.publish('depth', json.dumps(msg))


@click.command()
@click.option('--delay', '-d', type=str, default='4s')
@click.option('--num-locks', '-n', type=int, default=4)
def main(**kwargs):
    emitter = DepthEmitter(**kwargs)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda: exit(0))
    signal.signal(signal.SIGTERM, lambda: exit(0))
    main()
