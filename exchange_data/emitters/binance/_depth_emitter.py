#!/usr/bin/env python

import requests
from binance.client import Client
from binance.depthcache import DepthCache
from binance.exceptions import BinanceAPIException
from datetime import timedelta

from requests.exceptions import ProxyError, ReadTimeout, ConnectTimeout

from exchange_data import settings
from exchange_data.emitters import Messenger
from exchange_data.emitters.binance import BinanceUtils
from exchange_data.emitters.binance._depth_cache_manager import \
  NotifyingDepthCacheManager
from exchange_data.proxies_base import ProxiesBase
from exchange_data.utils import DateTimeUtils
from pytimeparse.timeparse import timeparse
from redis_collections import Set
from redlock import RedLock, RedLockError

import alog
import click
import json
import numpy as np
import os
import random
import signal


class ProxiedClient(Client, ProxiesBase):
  _proxies = None

  def __init__(self, proxies=None, **kwargs):
    self._proxies = proxies
    ProxiesBase.__init__(self, **kwargs)
    super().__init__(**kwargs)

  @property
  def proxies(self):
    if self._proxies:
      return self._proxies
    else:
      proxy = self.valid_proxies.random_sample()[0]

      return dict(
        http=proxy,
        https=proxy
      )

  def _init_session(self):
    session = requests.session()
    session.proxies.update(self.proxies)
    session.headers.update({'Accept': 'application/json',
                            'User-Agent': 'binance/python',
                            'X-MBX-APIKEY': self.API_KEY})
    return session


class DepthEmitter(Messenger, BinanceUtils):
  def __init__(self, lock_hold, interval, delay, num_symbol_take,
               num_locks=2, **kwargs):
    super().__init__(**kwargs)
    self.lock_hold = timeparse(lock_hold)
    self.interval = interval
    self.delay = timedelta(seconds=timeparse(delay))
    self.num_symbol_take = num_symbol_take
    self.num_locks = num_locks
    self.last_lock_id = 0
    self.lock = None
    self.caches = {}
    self.cache_queue = []

    alog.info('### initialized ###')

    self.on(self.interval, self.check_queues)

    self.check_queues()

  @property
  def symbols_queue(self):
    return Set(key='symbols_queue', redis=self.redis_client)

  @property
  def remove_symbols_queue(self):
    return Set(key='remove_symbols_queue',
               redis=self.redis_client)

  def check_queues(self, timestamp=None):
    self.purge()

    alog.info('### check queues ###')

    for symbol in self.remove_symbols_queue:
      self.remove_cache(symbol)

    if len(self.symbols_queue) > 0:
      for i in range(0, self.num_symbol_take):
        self.add_next_cache()

  def add_next_cache(self):
    try:
      queue_len = len(self.symbols_queue)
      next_ix = random.randrange(0, queue_len - 1)
      symbol = list(self.symbols_queue)[next_ix]
      self.symbols_queue.remove(symbol)

      if symbol:
        alog.info(f'## take next {symbol} ##')
        self.add_cache(symbol)

    except RedLockError:
      pass

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

  @property
  def client(self):
    return ProxiedClient()

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

    self.caches[symbol] = NotifyingDepthCacheManager(
      callback=self.message,
      client=self.client,
      limit=100000,
      redis_client=self.redis_client,
      refresh_interval=60 * 60 * 2,
      symbol=symbol,
      lock_hold=self.lock_hold
    )

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

      if depthCache.last_publish_time is None or \
        depthCache.last_publish_time < DateTimeUtils.now() - self.delay:
        depthCache.last_publish_time = depthCache.update_time

        self.publish('depth', json.dumps(msg))
        self.publish('symbol_timeout', json.dumps(dict(
          symbol=symbol,
          symbol_host=NotifyingDepthCacheManager._symbol_hostname(symbol)
        )))

  def start(self):
    self.sub([self.interval, 'remove_symbol'])


@click.command()
@click.option('--delay', '-d', type=str, default='15s')
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
