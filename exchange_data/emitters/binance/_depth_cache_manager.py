from binance.depthcache import DepthCacheManager
from exchange_data import settings
from exchange_data.emitters.binance import BinanceUtils
from exchange_data.emitters.binance.proxied_client import ProxiedClient
from exchange_data.utils import DateTimeUtils
from pytimeparse.timeparse import timeparse
from redis import Redis
from redis_collections import Set
from redlock import RedLock
from requests.exceptions import ProxyError, ConnectTimeout, ReadTimeout

import alog


class NotifyingDepthCacheManager(DepthCacheManager, BinanceUtils):
    _symbol = None
    last_publish_time = None

    def __init__(self, symbol, lock_hold, init_retry=3, **kwargs):
        self.lock_hold = lock_hold
        self.init_retry = init_retry
        super().__init__(symbol=symbol, client=ProxiedClient(), **kwargs)
        self.redis_client = Redis(host=settings.REDIS_HOST)
        self.symbol_hosts.add((symbol, self.symbol_hostname))
        self.last_publish_time = None
        self.created_at = DateTimeUtils.now()

    @property
    def symbol_hosts(self):
        return Set(key='symbol_hosts', redis=self.redis_client)

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
        )], retry_delay=1000 * 1, retry_times=24, ttl=timeparse('10s') * 1000)

    def _init_cache(self):
        try:
            if self.init_retry > 0:
                alog.info(f'## init retry {self.init_retry}')
                self.init_retry -= 1
                return super()._init_cache()
        except (ConnectTimeout, ConnectionError, ProxyError, ReadTimeout) as e:
            return self._init_cache()

    def close(self, **kwargs):
        try:
            self.symbol_hosts.remove((self._symbol, self.symbol_hostname))
        except KeyError:
            pass

        self.redis_client.close()

        kwargs['close_socket'] = True
        super().close(**kwargs)

    def __del__(self):
        alog.info(f'### delete cache ###')
