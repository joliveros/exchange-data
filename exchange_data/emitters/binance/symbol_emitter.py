#!/usr/bin/env python

import gc

from binance.depthcache import DepthCache
from binance.exceptions import BinanceAPIException
from datetime import timedelta

from unicorn_binance_websocket_api import BinanceWebSocketApiManager

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


class SymbolEmitter(Messenger, BinanceUtils, BinanceWebSocketApiManager):
    create_at = None
    last_queue_check = None

    def __init__(
        self,
        lock_hold,
        interval,
        delay,
        num_symbol_take,
        num_locks=2,
        **kwargs
    ):
        super().__init__(exchange="binance.com", **kwargs)
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

        self.create_stream(['depth'], self.symbols)

        while True:
            data = self.pop_stream_data_from_stream_buffer()
            if data:
                self.publish('depth', data)



@click.command()
@click.option('--delay', '-d', type=str, default='15s')
@click.option('--lock-hold', '-l', type=str, default='3s')
@click.option('--interval', '-i', type=str, default='5m')
@click.option('--num-symbol-take', '-n', type=int, default=4)
def main(**kwargs):
    emitter = SymbolEmitter(**kwargs)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda *args: exit(0))
    signal.signal(signal.SIGTERM, lambda *args: exit(0))
    main()
