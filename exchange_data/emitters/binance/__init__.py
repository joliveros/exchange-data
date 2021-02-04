from datetime import datetime

import docker
import math
from binance.exceptions import BinanceAPIException
from cached_property import cached_property_with_ttl, cached_property
from dateutil.tz import tz
from pytimeparse.timeparse import timeparse
from redis_collections import Set
from redlock import RedLock, RedLockError

from exchange_data import settings
from exchange_data.emitters.binance.proxied_client import ProxiedClient
from exchange_data.utils import DateTimeUtils

import alog
import re
import time


class BinanceUtils(object):
    limit = 0
    max_symbols = None
    last_start = None

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

    @cached_property
    def client(self):
        return ProxiedClient()

    @cached_property
    def queued_symbols(self):
        return Set(key='queued_symbols', redis=self.redis_client)

    @cached_property_with_ttl(ttl=60 * 10)
    def symbols(self):
        return self.get_symbols()

    def update_queued_symbols(self, prefix):
        try:
            with self.take_lock(prefix=f'{prefix}_symbol_update',
                                retry_times=1,
                                retry_delay=300):
                with self.take_lock(prefix=prefix):
                    alog.info('try to update queued symbols')
                    self.queued_symbols.clear()
                    self.queued_symbols.update(set(self.symbols))
                    alog.info(self.queued_symbols)

        except RedLockError:
            pass

        time.sleep(30)

    def get_symbols(self):
        try:
            return self._get_symbols()
        except Exception as e:
            return self.get_symbols()

    def _get_symbols(self):
        exchange_info = self.client.get_exchange_info()

        symbols = [symbol for symbol in exchange_info['symbols']
                   if symbol['status'] == 'TRADING']

        symbol_names = [symbol['symbol'] for symbol in symbols if symbol[
            'symbol']]

        return symbol_names

    def sleep_during_embargo(self, e):
        if e.status_code == 418:
            embargo_timestamp = int(re.search('\d+', e.message)[0])
            embargo_timestamp = datetime.fromtimestamp(embargo_timestamp / 1000) \
                .replace(tzinfo=tz.tzlocal())
            alog.info(f'banned until {embargo_timestamp}')
            diff = embargo_timestamp - DateTimeUtils.now()

            sleep_seconds = diff.total_seconds()

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    def take_symbols(self, *args, workers, prefix='', **kwargs):
        kwargs['workers'] = workers

        if self.max_symbols is None:
            with self.queued_symbols as queued_symbols:
                self.max_symbols = int(math.ceil((len(self.symbols) / workers)))
                alog.info(self.max_symbols)

        try:
            while len(self.queued_symbols) > 0 and len(self.depth_symbols) < self.max_symbols:
                if self.limit > 0:
                    if len(self.depth_symbols) > self.limit:
                        break
                with self.take_lock(prefix):
                    self._take_symbols(*args, **kwargs)
                time.sleep(2)
        except RedLockError as e:
            alog.info(e)
            self.take_symbols(*args, prefix=prefix, **kwargs)

    def _take_symbols(self, *args, workers=8, **kwargs):
        alog.info('### take symbols ##')
        with self.queued_symbols as queued_symbols:
            len_queued_symbols = len(queued_symbols)
            take_count = int(math.ceil(self.max_symbols / 2))

            if len_queued_symbols < take_count:
                take_count = len_queued_symbols

            symbols = queued_symbols.random_sample(k=take_count)

            for symbol in symbols:
                alog.info(f'### taking {symbol} ###')

                self.remove_symbol_queue(symbol)

                if symbol in queued_symbols:
                    self.remove_symbol_queue(symbol)

                self.depth_symbols.add(symbol)

            alog.info(len(self.depth_symbols))

    def remove_symbol_queue(self, symbol):
        try:
            self.queued_symbols.remove(symbol)
        except KeyError:
            pass

    def take_lock(self, prefix='', retry_times=60*60, retry_delay=1000):
        lock_name = f'{prefix}_take_lock'

        lock = RedLock(lock_name, [dict(
                host=settings.REDIS_HOST,
                db=0
            )],
            retry_delay=retry_delay,
            retry_times=retry_times,
            ttl=timeparse('2m') * 1000
        )

        alog.info(lock_name)

        return lock


class ExceededLagException(Exception):
    pass
