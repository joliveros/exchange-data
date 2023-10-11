import logging
from datetime import datetime


from cached_property import cached_property_with_ttl, cached_property
from dateutil.tz import tz
from pytimeparse.timeparse import timeparse
from redis import Redis
from redis_cache import RedisCache
from redis_collections import Set
from redlock import RedLock, RedLockError
from retry import retry
import math

from exchange_data import settings
from exchange_data.emitters.binance.proxied_client import ProxiedClient
from exchange_data.utils import DateTimeUtils

import alog
import re
import time

cache = RedisCache(redis_client=Redis(host=settings.REDIS_HOST))


class BinanceUtils(object):
    limit = 0
    max_symbols = None
    last_start = None

    def __init__(
        self, futures, symbols=None, log_requests=False, symbol_filter="BNB", **kwargs
    ):
        super().__init__(**kwargs)
        self.futures = futures
        self.symbol_filter = symbol_filter
        self._symbols = symbols

        if log_requests:
            self.log_requests()

    @cached_property
    def client(self):
        return ProxiedClient()

    @property
    def symbol_info(self):
        return [
            info
            for info in self.get_exchange_info["symbols"]
            if info["symbol"] == self.symbol
        ][0]

    @cached_property
    def queued_symbols(self):
        return Set(key="queued_symbols", redis=self.redis_client)

    @property
    def symbols(self):
        return self._symbols
        # symbols = self._get_symbols()
        #
        # if self.symbol_filter:
        #     return [symbol for symbol in symbols if symbol.endswith(self.symbol_filter)]
        # else:
        #     return symbols

    def update_queued_symbols(self, prefix):
        try:
            with self.take_lock(
                prefix=f"{prefix}_symbol_update", retry_times=1, retry_delay=300
            ):
                with self.take_lock(prefix=prefix):
                    alog.info("try to update queued symbols")
                    self.queued_symbols.clear()
                    self.queued_symbols.update(set(self.symbols))

                    alog.info(self.queued_symbols)
                    alog.info(f"number of symbols {len(self.queued_symbols)}")

        except RedLockError:
            pass

        time.sleep(30)

    def get_symbols(self):
        try:
            return self._get_symbols()
        except Exception as e:
            alog.info(e)
            return self.get_symbols()

    @retry(Exception, tries=100, delay=0.5)
    def _retry_get_exchange_info(self):
        return self._get_exchange_info()

    def _get_exchange_info(self):
        if self.futures:
            return BinanceUtils.futures_exchange_info()
        else:
            return BinanceUtils.exchange_info()

    @staticmethod
    @cache.cache(ttl=60 * 60)
    def futures_exchange_info():
        return ProxiedClient().futures_exchange_info()

    @staticmethod
    @cache.cache(ttl=60 * 60)
    def exchange_info():
        return ProxiedClient().get_exchange_info()

    @property
    def get_exchange_info(self):
        return self._retry_get_exchange_info()

    @property
    def lot_size(self):
        return [
            filter
            for filter in self.symbol_info["filters"]
            if filter["filterType"] == "LOT_SIZE"
        ][0]

    @property
    def price_filter(self):
        return [
            filter
            for filter in self.symbol_info["filters"]
            if filter["filterType"] == "PRICE_FILTER"
        ][0]

    @property
    def step_size(self):
        return float(self.lot_size["stepSize"])

    @property
    def precision(self):
        return self.symbol_info.get(
            "quoteAssetPrecision", None
        ) or self.symbol_info.get("quotePrecision", None)

    @property
    def tick_size(self):
        return float(self.price_filter["tickSize"])

    @property
    def bracket(self):
        return [
            bracket
            for bracket in self.leverage_brackets()
            if bracket["symbol"] == self.symbol
        ][0]

    @property
    def max_leverage(self):
        if self.futures:
            return max(
                [bracket["initialLeverage"] for bracket in self.bracket["brackets"]]
            )
        else:
            raise Exception("only available for futures.")

    def _get_symbols(self):
        exchange_info = self.get_exchange_info

        symbols = [
            symbol
            for symbol in exchange_info["symbols"]
            if symbol["status"] == "TRADING"
        ]

        symbol_names = [symbol["symbol"] for symbol in symbols if symbol["symbol"]]

        return symbol_names

    def sleep_during_embargo(self, e):
        if e.status_code == 418:
            embargo_timestamp = int(re.search("\d+", e.message)[0])
            embargo_timestamp = datetime.fromtimestamp(
                embargo_timestamp / 1000
            ).replace(tzinfo=tz.tzlocal())
            alog.info(f"banned until {embargo_timestamp}")
            diff = embargo_timestamp - DateTimeUtils.now()

            sleep_seconds = diff.total_seconds()

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    def take_symbols(self, *args, workers, prefix="", **kwargs):
        kwargs["workers"] = workers

        if self.max_symbols is None:
            with self.queued_symbols as queued_symbols:
                self.max_symbols = int(math.ceil((len(self.symbols) / workers)))
                alog.info(self.max_symbols)

        try:
            while (
                len(self.queued_symbols) > 0
                and len(self.depth_symbols) < self.max_symbols
            ):
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
        alog.info("### take symbols ##")
        with self.queued_symbols as queued_symbols:
            len_queued_symbols = len(queued_symbols)
            take_count = int(math.ceil(self.max_symbols / 2))

            if len_queued_symbols < take_count:
                take_count = len_queued_symbols

            symbols = queued_symbols.random_sample(k=take_count)

            for symbol in symbols:
                alog.info(f"### taking {symbol} ###")

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

    def take_lock(self, prefix="", retry_times=60 * 60, retry_delay=1000):
        lock_name = f"{prefix}_take_lock"

        lock = RedLock(
            lock_name,
            [dict(host=settings.REDIS_HOST, db=0)],
            retry_delay=retry_delay,
            retry_times=retry_times,
            ttl=timeparse("2m") * 1000,
        )

        alog.info(lock_name)

        return lock

    def log_requests(self):
        loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
        logger = [logger for logger in loggers if logger.name == "urllib3"][0]
        logger.setLevel(logging.DEBUG)

    def channel_suffix(self, channel):
        if self.futures:
            return f"{channel}_futures"
        else:
            return channel


class ExceededLagException(Exception):
    pass


def truncate(n, decimals=0):
    multiplier = 10**decimals
    return int(n * multiplier) / multiplier
