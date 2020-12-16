import time

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

    def __init__(self, symbol, lock_hold, init_retry=2, **kwargs):
        self.lock_hold = lock_hold
        self._init_retry = init_retry
        self.init_retry = init_retry
        super().__init__(symbol=symbol, client=ProxiedClient(), **kwargs)
        self.redis_client = Redis(host=settings.REDIS_HOST)
        self.symbol_hosts.add((symbol, self.symbol_hostname))
        self.last_publish_time = None
        self.created_at = DateTimeUtils.now()

        self.close()

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

    def get_orderbook(self, **kwargs):
        try:
            if self.init_retry > 0:
                alog.info(f'## init retry {self._symbol} {self.init_retry}')
                self.init_retry -= 1
                res = self._client.get_order_book(**kwargs)
                self.init_retry = self._init_retry
                return res
            else:
                raise Exception()
        except (ConnectTimeout, ConnectionError, ProxyError, ReadTimeout) as e:
            return self.get_orderbook(**kwargs)

    def _init_cache(self):
        """Initialise the depth cache calling REST endpoint

        :return:
        """

        try:
            res = self.get_orderbook(symbol=self._symbol, limit=self._limit)
        except Exception as e:
            return

        self._last_update_id = None
        self._depth_message_buffer = []

        # process bid and asks from the order book
        for bid in res['bids']:
            self._depth_cache.add_bid(bid)
        for ask in res['asks']:
            self._depth_cache.add_ask(ask)

        # set first update id
        self._last_update_id = res['lastUpdateId']

        # set a time to refresh the depth cache
        if self._refresh_interval:
            self._refresh_time = int(time.time()) + self._refresh_interval

        # Apply any updates from the websocket
        for msg in self._depth_message_buffer:
            self._process_depth_message(msg, buffer=True)

        # clear the depth buffer
        self._depth_message_buffer = []

    def _depth_event(self, msg):
        try:
            super()._depth_event(msg)
        except Exception as e:
            alog.info(e)
            self.close()

    def close(self, **kwargs):
        try:
            self.symbol_hosts.remove((self._symbol, self.symbol_hostname))
        except KeyError:
            pass

        self.redis_client.close()

        kwargs['close_socket'] = True

        super().close(**kwargs)

        alog.info(f'## closed {self._symbol} ##')

    def __del__(self):
        alog.info(f'### delete cache ###')
