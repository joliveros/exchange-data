#!/usr/bin/env python
import gc
import threading

from binance.depthcache import DepthCache
from binance.exceptions import BinanceAPIException
from datetime import timedelta
from exchange_data import settings
from exchange_data.emitters import Messenger
from exchange_data.emitters.binance import BinanceUtils
from exchange_data.emitters.binance._depth_cache_manager import \
    NotifyingDepthCacheManager
from exchange_data.utils import DateTimeUtils
from pytimeparse.timeparse import timeparse
from redis_collections import Set
from redlock import RedLock, RedLockError
from requests.exceptions import ProxyError, ReadTimeout, ConnectTimeout

import alog
import click
import json
import numpy as np
import os
import random
import signal


class DepthEmitter(Messenger, BinanceUtils):
    create_at = None

    def __init__(self, lock_hold, interval, delay, num_symbol_take,
                 max_life,
                 num_locks=2, **kwargs):
        super().__init__(**kwargs)
        self.max_life = timeparse(max_life)
        self.lock_hold = timeparse(lock_hold)
        self.interval = interval
        self.delay = timedelta(seconds=timeparse(delay))
        self.num_symbol_take = num_symbol_take
        self.num_locks = num_locks
        self.last_lock_id = 0
        self.lock = None
        self.caches = {}
        self.cache_queue = []
        self.create_at = DateTimeUtils.now()

        alog.info('### initialized ###')

        self.on(self.interval, self.check_queues)

        self.check_queues()

    @property
    def symbol_hosts(self):
        return Set(key='symbol_hosts', redis=self.redis_client)

    @property
    def max_caches(self):
        hosts = set()

        for symbol, symbol_host in self.symbol_hosts:
            hosts.add(symbol_host.split('_')[-1])

        if len(hosts) == 0:
            return 0
        else:
            return int(len(self.symbols) / len(hosts))

    @property
    def symbols_queue(self):
        return Set(key='symbols_queue', redis=self.redis_client)

    @property
    def remove_symbols_queue(self):
        return Set(key='remove_symbols_queue',
                   redis=self.redis_client)

    @property
    def time_since_created(self):
        return (DateTimeUtils.now() - self.create_at).total_seconds()

    def check_queues(self, timestamp=None):
        alog.info((self.time_since_created, self.max_life))

        if self.time_since_created > self.max_life:
            # self.requeue_symbols()
            self.exit()

        self.purge()

        alog.info('### check queues ###')
        alog.info(f'num threads {len(threading.enumerate())}')

        for symbol in self.remove_symbols_queue:
            self.remove_cache(symbol)

        alog.info('started gc')
        gc.collect()
        alog.info('end gc')

        alog.info((len(self.caches), self.max_caches))

        if len(self.symbols_queue) > 0 and self.should_take:
            for i in range(0, self.num_symbol_take):
                self.add_next_cache()

    @property
    def should_take(self):
        should_take = False

        if self.max_caches > 0 and len(self.caches) < self.max_caches:
            should_take = True
        elif self.max_caches == 0:
            should_take = True

        return should_take

    def requeue_symbols(self):
        alog.info('## requeue symbols ##')

        _symbol_hosts = set([s for s in self.symbol_hosts])

        for symbol, cache in self.caches.items():
            self.symbols_queue.add(symbol)
            self.symbol_hosts.remove((symbol,
                                      NotifyingDepthCacheManager._symbol_hostname(
                                          symbol)))

    def add_next_cache(self):
        alog.info('attempt add next cache')
        try:
            queue_len = len(self.symbols_queue)

            if queue_len > 0:
                if queue_len == 1:
                    next_ix = 0
                else:
                    next_ix = random.randrange(0, queue_len - 1)
                symbol = list(self.symbols_queue)[next_ix]

                try:
                    self.symbols_queue.remove(symbol)
                except KeyError as e:
                    pass

                if symbol:
                    alog.info(f'## take next {symbol} ##')
                    self.add_cache(symbol)

        except RedLockError as e:
            alog.info(e)

    def purge(self):
        caches: [NotifyingDepthCacheManager] = list(self.caches.values())

        for cache in caches:
            _cache: DepthCache = cache._depth_cache
            dcm = self.caches[_cache.symbol]

            alog.info(f'last publish time {dcm._symbol} {dcm.last_publish_time}')

            if dcm.last_publish_time:
                if dcm.last_publish_time < DateTimeUtils.now() \
                   - timedelta(seconds=60):
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

    def take_lock(self):
        lock_name = f'take_lock'

        lock = RedLock(lock_name, [dict(
            host=settings.REDIS_HOST,
            db=0
        )], retry_delay=1000, retry_times=60 * 60, ttl=timeparse('30s') * 1000)

        alog.info(lock_name)

        return lock

    def add_cache_lock(self, symbol):
        lock_name = f'add_cache_lock_{symbol}'

        alog.info(lock_name)

        return RedLock(lock_name, [dict(
            host=settings.REDIS_HOST,
            db=0
        )], retry_delay=200, retry_times=1, ttl=timeparse('30s') * 1000)

    def add_cache(self, symbol):
        try:
            with self.add_cache_lock(symbol):
                self._add_cache(symbol)

        except BinanceAPIException as e:
            alog.info(alog.pformat(vars(e)))
            self.sleep_during_embargo(e)

        except (ConnectTimeout, ConnectionError, ProxyError, ReadTimeout) as e:
            alog.info(e)
            self.symbols_queue.add(symbol)

    def _add_cache(self, symbol):
        if symbol in self.caches.keys():
            self.remove_cache(NotifyingDepthCacheManager._symbol_hostname(
                symbol))

        dcm = NotifyingDepthCacheManager(
            callback=self.message,
            limit=100000,
            refresh_interval=60 * 60 * 2,
            symbol=symbol,
            lock_hold=self.lock_hold
        )

        self.caches[symbol] = dcm

        alog.info(f'### cache added {symbol}###')

    def exit(self):
        os.kill(os.getpid(), signal.SIGKILL)

    def message(self, depthCache: DepthCache):
        if depthCache:
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

            if depthCache.symbol in self.caches:
                dcm = self.caches[depthCache.symbol]
                if dcm.last_publish_time is None or \
                    dcm.last_publish_time < DateTimeUtils.now() - self.delay:
                    dcm.last_publish_time = DateTimeUtils.now()

                    self.publish('depth', json.dumps(msg))
                    self.publish('symbol_timeout', json.dumps(dict(
                        symbol=symbol,
                        symbol_host=NotifyingDepthCacheManager._symbol_hostname(
                            symbol)
                    )))

    def start(self):
        self.sub([self.interval, 'remove_symbol'])


@click.command()
@click.option('--delay', '-d', type=str, default='15s')
@click.option('--max-life', '-m', type=str, default='2h')
@click.option('--lock-hold', '-l', type=str, default='3s')
@click.option('--interval', '-i', type=str, default='5m')
@click.option('--num-symbol-take', '-n', type=int, default=4)
def main(**kwargs):
    emitter = DepthEmitter(**kwargs)
    emitter.start()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda *args: exit(0))
    signal.signal(signal.SIGTERM, lambda *args: exit(0))
    main()
