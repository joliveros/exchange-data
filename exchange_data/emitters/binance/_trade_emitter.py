#!/usr/bin/env python
import logging

import txaio
from binance.client import Client
from binance.exceptions import BinanceAPIException
from binance.websockets import BinanceSocketManager
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

# txaio.start_logging(level='debug')
# txaio.set_global_log_level('debug')


class TradeEmitterLock(RedLock):
    def __init__(self, symbol, **kwargs):
        lock_name = f'trade_emitter_lock_{symbol}'

        connection_details = [dict(
            host=settings.REDIS_HOST,
            db=0
        )]

        super().__init__(resource=lock_name,
                         connection_details=connection_details,
                         retry_delay=200,
                         retry_times=3,
                         ttl=timeparse('5m') * 1000,
                         **kwargs)


class TradeSocketManager(BinanceSocketManager):
    def __init__(self, symbol, redis_client, callback, delay,
                 **kwargs):

        super().__init__(user_timeout=timeparse('5m'), **kwargs)
        self.symbol = symbol
        self.lock = TradeEmitterLock(symbol)
        lock = self.lock.acquire()

        if not lock:
            raise RedLockError(f'Unable to acquire lock {self.symbol}')

        self.symbol_hosts = Set(key='trade_symbol_hosts', redis=redis_client)
        self.symbol_hosts.add((symbol, self.symbol_hostname))
        self.last_publish_time = DateTimeUtils.now() - delay
        self.last_message = None
        self.callback = callback

        def message(*args):
            self.last_message = DateTimeUtils.now()
            args += (self,)
            self.callback(*args)

        conn = self.start_trade_socket(symbol, message)

        self.start()

    @property
    def symbol_hostname(self):
        return self._symbol_hostname(self.symbol)

    @staticmethod
    def _symbol_hostname(symbol):
        return f'{symbol}_{settings.HOSTNAME}'

    def close(self, **kwargs):
        self.lock.release()
        try:
            self.symbol_hosts.remove((self.symbol, self.symbol_hostname))
        except KeyError:
            pass

        super().close(**kwargs)

        alog.info(f'### successfully closed socket {self.symbol}###')


class TradeEmitter(Messenger):
    def __init__(self, delay, num_take_symbols, **kwargs):
        super().__init__(**kwargs)

        self.num_take_symbols = num_take_symbols
        self.delay = timedelta(seconds=timeparse(delay))
        self.last_lock_id = 0
        self.lock = None
        self.sockets = {}

        alog.info('### initialized ###')

        self.on('5s', self.check_queues)
        self.check_queues()

    @property
    def remove_symbols_queue(self):
        return Set(key='remove_trade_symbols_queue',
            redis=self.redis_client)

    @property
    def symbols_queue(self):
        return Set(key='trade_symbols_queue',
            redis=self.redis_client)

    def check_queues(self, timestamp=None):
        alog.info('### check queues ###')

        for symbol in self.remove_symbols_queue:
            self.remove_socket(symbol)

        if len(self.symbols_queue) > 0:
            alog.info(self.num_take_symbols)
            for i in range(0, self.num_take_symbols):
                self.add_next_trade_socket(self.symbols_queue)

    def add_next_trade_socket(self, queue):
        symbol = self.next_symbol(queue)

        if symbol:
            alog.info(f'## take next {symbol} ##')
            self.add_trade_socket(symbol)

    def next_symbol(self, *args):
        try:
            return self._next_symbol(*args)
        except RedLockError as e:
            time.sleep(0.1)
            return self.next_symbol(*args)

    def _next_symbol(self, queue):
        with self.take_lock():
            symbol = queue.pop()

        return symbol

    def purge(self):
        alog.info('### purge ###')
        sockets: [TradeSocketManager] = list(self.sockets.values())

        for socket in sockets:
            if socket.last_publish_time:
                if socket.last_publish_time < DateTimeUtils.now() - timedelta(
                    seconds=60):
                    self.remove_socket(socket.symbol_hostname)

    @property
    def symbol_hostnames(self):
        return [s.symbol_hostname for s in self.sockets.values()]

    def remove_socket(self, symbol_host):
        if '_' not in symbol_host:
            raise Exception('Not as symbol-hostname')

        for socket in self.symbol_hostnames:
            if socket == symbol_host:
                alog.info(f'### remove {symbol_host} ###')

                if socket in self.remove_symbols_queue:
                    self.remove_symbols_queue.remove(socket)

                sockets = [socket for socket in self.sockets.values() if
                          socket.symbol_hostname == symbol_host]

                if len(sockets) > 0:
                    _socket = sockets[0]
                    self.sockets[_socket.symbol].close()
                    del self.sockets[_socket.symbol]

    @cached_property
    def client(self):
        return Client()

    def take_lock(self):
        lock_name = 'take_lock'

        alog.info(lock_name)

        return RedLock(lock_name, [dict(
            host=settings.REDIS_HOST,
            db=0
        )], retry_delay=200, retry_times=1000, ttl=timeparse('1m') * 1000)

    def add_trade_socket(self, symbol):
        try:
            self._add_cache(symbol)

        except BinanceAPIException as e:
            alog.info(alog.pformat(vars(e)))

            if e.status_code == 418:
                time.sleep(30)

        except RedLockError as e:
            alog.info(e)

    def _add_cache(self, symbol):
        if symbol in self.sockets.keys():
            self.remove_socket(TradeSocketManager._symbol_hostname(
                symbol))

        self.sockets[symbol] = TradeSocketManager(
            delay=self.delay,
            callback=self.message,
            client=self.client,
            redis_client=self.redis_client,
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

    def message(self, data, socket):
        msg = dict(
            price=float(data['p']),
            quantity=float(data['q']),
            symbol=data['s'],
            timestamp=str(DateTimeUtils.parse_db_timestamp(data['T'])),
        )

        self.publish('trade', json.dumps(msg))
        self.publish('trade_symbol_timeout', json.dumps(dict(
            symbol=socket.symbol,
            symbol_host=socket.symbol_hostname
        )))

    def start(self):
        self.sub(['2s', '5s', 'remove_symbol'])


@click.command()
@click.option('--delay', '-d', type=str, default='15s')
@click.option('--num_take_symbols', '-n', type=int, default=4)
def main(**kwargs):
    emitter = TradeEmitter(**kwargs)
    emitter.start()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda: exit(0))
    signal.signal(signal.SIGTERM, lambda: exit(0))
    main()
