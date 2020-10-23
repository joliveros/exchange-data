#!/usr/bin/env python

from binance.client import Client
from binance.depthcache import DepthCacheManager, DepthCache
from binance.exceptions import BinanceAPIException
from cached_property import cached_property
from datetime import timedelta
from exchange_data import settings
from exchange_data.emitters import Messenger
from exchange_data.utils import DateTimeUtils
from pytimeparse.timeparse import timeparse
from redis_collections import Set
from redlock import RedLock, RedLockError

import alog
import click
import json
import numpy as np
import os
import signal
import time


class NotifyingDepthCacheManager(DepthCacheManager):
    def __init__(self, symbol, redis_client, **kwargs):
        super().__init__(symbol=symbol, **kwargs)
        self.symbol_hosts = Set(key='symbol_hosts', redis=redis_client)
        self.symbol_hosts.add((symbol, self.symbol_hostname))

    @property
    def symbol_hostname(self):
        return self._symbol_hostname(self._symbol)

    @staticmethod
    def _symbol_hostname(symbol):
        return f'{symbol}_{settings.HOSTNAME}'

    def close(self, **kwargs):
        try:
            self.symbol_hosts.remove((self._symbol, self.symbol_hostname))
        except KeyError:
            pass
        kwargs['close_socket'] = True
        super().close(**kwargs)


class DepthEmitter(Messenger):
    def __init__(self, delay, num_symbol_take, num_locks=2, **kwargs):
        super().__init__(**kwargs)

        self.delay = timedelta(seconds=timeparse(delay))
        self.num_symbol_take = num_symbol_take
        self.num_locks = num_locks
        self.last_lock_id = 0
        self.lock = None
        self.symbols_queue = Set(key='symbols_queue', redis=self.redis_client)
        self.remove_symbols_queue = Set(key='remove_symbols_queue',
                                    redis=self.redis_client)
        self.caches = {}
        self.cache_queue = []

        alog.info('### initialized ###')

        self.on('5s', self.check_queues)
        self.check_queues()

    def check_queues(self, timestamp=None):
        self.purge()

        alog.info('### check queues ###')
        caches = list(self.caches.keys())

        for symbol in self.remove_symbols_queue:
            self.remove_cache(symbol)

        if len(self.symbols_queue) > 0:
            for i in range(0, self.num_symbol_take):
                self.add_next_cache()

    def add_next_cache(self):
        symbol = self.symbols_queue.pop()

        if symbol:
            alog.info(f'## take next {symbol} ##')
            self.add_cache(symbol)

    def purge(self):
        caches: [NotifyingDepthCacheManager] = list(self.caches.values())

        for cache in caches:
            _cache: DepthCache = cache._depth_cache

            if _cache.last_publish_time:
                if _cache.last_publish_time < DateTimeUtils.now() - timedelta(
                    seconds=60):
                    self.remove_cache(cache.symbol_hostname)

    @property
    def symbol_hostnames(self):
        return [s.symbol_hostname for s in self.caches.values()]

    def remove_cache(self, symbol_host):
        if '_' not in symbol_host:
            raise Exception('Not as symbol-hostname')

        for cache in self.symbol_hostnames:
            if cache == symbol_host:
                alog.info(f'### remove {symbol_host} ###')

                if cache in self.remove_symbols_queue:
                    self.remove_symbols_queue.remove(cache)

                caches = [cache for cache in self.caches.values() if
                          cache.symbol_hostname == symbol_host]
                if len(caches) > 0:
                    _cache = caches[0]
                    self.caches[_cache._symbol].close()
                    del self.caches[_cache._symbol]

    @cached_property
    def client(self):
        return Client()

    def add_cache_lock(self, symbol):
        lock_name = f'add_cache_lock_{symbol}'

        alog.info(lock_name)

        return RedLock(lock_name, [dict(
            host=settings.REDIS_HOST,
            db=0
        )], retry_delay=200, retry_times=3, ttl=timeparse('3m') * 1000)

    def add_cache(self, symbol):
        try:
            with self.add_cache_lock(symbol):
                self._add_cache(symbol)

        except BinanceAPIException as e:
            alog.info(alog.pformat(vars(e)))

            if e.status_code == 418:
                time.sleep(30)

        except RedLockError as e:
            alog.info(e)
            pass

    def _add_cache(self, symbol):
        if symbol in self.caches.keys():
            self.remove_cache(NotifyingDepthCacheManager._symbol_hostname(
                symbol))

        self.caches[symbol] = NotifyingDepthCacheManager(
            callback=self.message,
            client=self.client,
            limit=100,
            redis_client=self.redis_client,
            refresh_interval=0,
            symbol=symbol,
        )

        alog.info(f'### cache added {symbol}###')

    def get_symbols(self):
        exchange_info = self.client.get_exchange_info()

        symbols = [symbol for symbol in exchange_info['symbols']
                   if symbol['status'] == 'TRADING']

        symbol_names = [symbol['symbol'] for symbol in symbols if symbol[
            'symbol']]

        return symbol_names

    def exit(self):
        os.kill(os.getpid(), signal.SIGKILL)

    def message(self, depthCache: DepthCache):
        if depthCache is None:
            self.exit()
            raise Exception()

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

        msg = dict(
            symbol=symbol,
            depth=depth.tolist()
        )

        if depthCache.last_publish_time is None or \
            depthCache.last_publish_time < DateTimeUtils.now() - self.delay:
            depthCache.last_publish_time = depthCache.update_time

            self.publish('depth', json.dumps(msg))
            self.publish('symbol_timeout', json.dumps(dict(
                symbol=symbol,
                symbol_host=NotifyingDepthCacheManager._symbol_hostname(symbol)
            )))

    def start(self):
        self.sub(['5s', 'remove_symbol'])


@click.command()
@click.option('--delay', '-d', type=str, default='15s')
@click.option('--num-symbol-take', '-n', type=int, default=4)
def main(**kwargs):
    emitter = DepthEmitter(**kwargs)
    emitter.start()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda: exit(0))
    signal.signal(signal.SIGTERM, lambda: exit(0))
    main()
