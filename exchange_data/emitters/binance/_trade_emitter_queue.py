#!/usr/bin/env python

from binance.client import Client
from binance.exceptions import BinanceAPIException
from cached_property import cached_property
from collections import Counter
from datetime import timedelta, datetime
from exchange_data.emitters import Messenger
from exchange_data.utils import DateTimeUtils
from pytimeparse.timeparse import timeparse
from redis_collections import List, Dict, Set

import alog
import click
import signal


class TradeEmitterQueue(Messenger):
    def __init__(self, timeout_interval, **kwargs):
        self.timeout_seconds = timeparse(timeout_interval)
        self.timeout_interval = timedelta(seconds=self.timeout_seconds)

        super().__init__(**kwargs)

        self.symbols = self.get_symbols()

        self.remove_symbols_queue.clear()
        self.symbol_hosts.clear()
        self.symbol_hosts_timeout = {}

        self.on('30s', self.check_symbol_timeout)
        self.on('trade_symbol_timeout', self.symbol_timeout)

        self.check_symbol_timeout(None)

    @property
    def symbol_hosts(self):
        return Set(key='trade_symbol_hosts',
            redis=self.redis_client)

    @property
    def remove_symbols_queue(self):
        return Set(key='remove_trade_symbols_queue',
                   redis=self.redis_client)

    @property
    def symbols_queue(self):
        return Set(key='trade_symbols_queue',
                   redis=self.redis_client)

    def symbol_timeout(self, data):
        symbol = data['symbol']
        symbol_host = data['symbol_host']

        self.symbol_hosts_timeout[symbol_host] = DateTimeUtils.now()

        _symbol_host = (symbol, symbol_host)

        if _symbol_host not in self.symbol_hosts:
            self.symbol_hosts.add(_symbol_host)

        if symbol in self.remove_symbols_queue:
            self.remove_symbols_queue.remove(symbol)

        if symbol in self.symbols_queue:
            try:
                self.symbols_queue.remove(symbol)
            except KeyError as e:
                pass

    def remove_symbol(self, symbol_host):
        for _symbol, _symbol_host in self.symbol_hosts:
            if _symbol_host == symbol_host:
                alog.info(f'### queue {_symbol_host} for removal ###')

                try:
                    self.symbol_hosts.remove((_symbol, _symbol_host))
                except KeyError:
                    pass

                self.remove_symbols_queue.add(_symbol_host)

    def check_symbol_timeout(self, timestamp):
        self.add_symbols_to_queue()

        for symbol, symbol_host in self.symbol_hosts:
            if symbol_host in self.symbol_hosts_timeout:
                timestamp = self.symbol_hosts_timeout[symbol_host]
            else:
                timestamp = DateTimeUtils.now()
                self.symbol_hosts_timeout[symbol_host] = timestamp

            last_update = DateTimeUtils.now() - self.timeout_interval

            if timestamp < last_update:
                if symbol_host in self.symbol_hosts_timeout:
                    del self.symbol_hosts_timeout[symbol_host]
                self.remove_symbol(symbol_host)
                self.symbols_queue.add(symbol)

        self.remove_duplicates()

        alog.info(f'### running hosts {len(self.symbol_hosts)}')
        alog.info(f'### queue length {len(self.symbols_queue)}')

    def remove_duplicates(self):
        alog.info('### remove duplicates ###')
        # check for duplicates
        symbol_hosts = [(key.split('_')[0], key) for key in
                        self.symbol_hosts_timeout.keys()]
        symbols = [s[0] for s in symbol_hosts]
        duplicates = [symbol for symbol, count in Counter(
            symbols).items()
                      if count > 1]

        for symbol in duplicates:
            hosts = [symbol_host for symbol_host in symbol_hosts if
                     symbol == symbol_host[0]]

            try:
                hosts.pop()
            except:
                pass

            for host in hosts:
                self.remove_symbol(host)

    def add_symbols_to_queue(self):
        symbol_hosts = [s[0] for s in self.symbol_hosts]

        for symbol in self.symbols:
            if symbol not in symbol_hosts:
                self.symbols_queue.add(symbol)

    @cached_property
    def client(self):
        return Client()

    def get_symbols(self):
        try:
            return self._get_symbols()
        except BinanceAPIException as e:
            alog.info(e)

    def _get_symbols(self):
        exchange_info = self.client.get_exchange_info()

        symbols = [symbol for symbol in exchange_info['symbols']
                   if symbol['status'] == 'TRADING']

        symbol_names = [symbol['symbol'] for symbol in symbols if symbol[
            'symbol']]

        return symbol_names

    def start(self):
        self.sub(['30s', 'trade_symbol_timeout'])


@click.command()
@click.option('--timeout-interval', '-t', type=str, default='30s')
def main(**kwargs):
    emitter = TradeEmitterQueue(**kwargs)
    emitter.start()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda: exit(0))
    signal.signal(signal.SIGTERM, lambda: exit(0))
    main()
