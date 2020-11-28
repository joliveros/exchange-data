from binance.depthcache import DepthCacheManager
from binance.exceptions import BinanceAPIException
from exchange_data import settings
from exchange_data.emitters.binance import BinanceUtils
from pytimeparse.timeparse import timeparse
from redis_collections import Set
from redlock import RedLock

import alog
import time


class NotifyingDepthCacheManager(DepthCacheManager, BinanceUtils):
    def __init__(self, symbol, lock_hold, redis_client, **kwargs):
        self.lock_hold = lock_hold
        super().__init__(symbol=symbol, **kwargs)
        self.symbol_hosts = Set(key='symbol_hosts', redis=redis_client)
        self.symbol_hosts.add((symbol, self.symbol_hostname))


    @property
    def symbol_hostname(self):
        return self._symbol_hostname(self._symbol)

    @staticmethod
    def _symbol_hostname(symbol):
        return f'{symbol}_{settings.HOSTNAME}'

    def init_cache_lock(self):
        lock_name = f'init_cache_lock'

        alog.info(lock_name)

        return RedLock(lock_name, [dict(
            host=settings.REDIS_HOST,
            db=0
        )], retry_delay=1000, retry_times=60, ttl=timeparse('30s') * 1000)

    def _init_cache(self):
        with self.init_cache_lock():
            try:
                super()._init_cache()
                time.sleep(self.lock_hold)
            except BinanceAPIException as e:
                self.sleep_during_embargo(e)


    def close(self, **kwargs):
        try:
            self.symbol_hosts.remove((self._symbol, self.symbol_hostname))
        except KeyError:
            pass
        kwargs['close_socket'] = True
        super().close(**kwargs)
