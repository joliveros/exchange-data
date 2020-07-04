#!/usr/bin/env python
import json
import re
import sys
import time
from datetime import timedelta

import alog
import docker
import click
import signal

from binance.client import Client
from cached_property import cached_property
from docker.models.services import Service
from pytimeparse.timeparse import timeparse
from redis_collections import SyncableDict

from exchange_data.emitters import Messenger
from exchange_data.utils import DateTimeUtils


class BinanceDepthEmitters(Messenger):
    def __init__(self, stop_all, stop, restart, **kwargs):
        super().__init__(self, **kwargs)

        self.should_stop = stop
        self.should_restart = restart
        self.client = docker.from_env()
        self.image_name = 'registry.rubercubic.com:5001/exchange-data:latest'
        self.redis_ip_ix = 0
        self.timeout = timedelta(seconds=timeparse('20s'))
        self.symbols_update_delay = timedelta(seconds=timeparse('1h'))
        self.symbol_restart_delay = timedelta(seconds=timeparse('2s'))
        self.symbol_max_restart_delay = timedelta(seconds=timeparse('30s'))

        if stop_all:
            self.stop_all()

        return

        self.symbol_restart_queue = SyncableDict(redis=self.redis_client,
                                                 key='symbol_restart_queue')

        self.symbol_restart_queue.sync()

        self.symbol_timestamp = SyncableDict(redis=self.redis_client,
                                             key='symbol_timestamp')
        self.symbol_timestamp.sync()

        for symbol in self.symbols:
            # self.set_symbol_timestamp(dict(symbol=symbol))
            self.add_symbol_restart_queue(symbol)

        self.on('2s', self.queue_restart)
        self.on('2s', self.check_timeouts)
        self.on('30s', self.update_symbols)
        self.on('1m', self.sync)
        self.on('symbol_timeout', self.set_symbol_timestamp)

    def sync(self, timestamp):
        self.symbol_restart_queue.sync()
        self.symbol_timestamp.sync()
        alog.info('### sync ###')

    @property
    def last_symbols_update(self):
        last_symbols_update = self.redis_client.get('last_symbols_update')

        if last_symbols_update is None:
            return DateTimeUtils.now() - self.symbols_update_delay - \
                   timedelta(seconds=10)

        return DateTimeUtils.parse_datetime_str(last_symbols_update.decode())

    @property
    def last_restart(self):
        last_symbol_restart = self.redis_client.get('last_symbol_restart')

        if last_symbol_restart is None:
            return DateTimeUtils.now() - self.symbol_restart_delay

        return DateTimeUtils.parse_datetime_str(last_symbol_restart.decode())

    def queue_restart(self, timestamp):
        if self.last_restart < DateTimeUtils.now() - self.symbol_restart_delay:
            self.restart_symbol(timedelta)

    def restart_symbol(self, timestamp):
        queue = self.symbol_restart_queue

        last_restart = DateTimeUtils.now() - self.symbol_max_restart_delay

        symbols = [(symbol[-1], symbol[0]) for symbol in queue.items()
                   if symbol[-1] is None or symbol[-1] < last_restart]

        symbols = sorted(symbols, reverse=True)

        alog.info(f'### queued restarts {len(symbols)} ###')

        if len(symbols) > 0:
            symbol = symbols.pop()[-1]

            alog.info(f'### restart {symbol} ###')
            self.redis_client.set('last_symbol_restart',
                                  str(DateTimeUtils.now()))

            self.symbol_restart_queue[symbol] = DateTimeUtils.now()

            self.restart(symbol)

    @cached_property
    def bnb_client(self):
        return Client()

    def set_symbol_timestamp(self, data):
        symbol = data['symbol']
        now = DateTimeUtils.now()
        self.symbol_timestamp[symbol] = now
        self.symbol_restart_queue[symbol] = now

    @property
    def symbols(self):
        symbols = self.redis_client.get('symbols')
        if symbols:
            return json.loads(symbols)
        else:
            return []

    def update_symbols(self, timestamp):
        if len(self.symbols) == 0:
            self._update_symbols()

        if self.last_symbols_update < DateTimeUtils.now() - self.symbols_update_delay:
            self._update_symbols()

    def _update_symbols(self):
        self.redis_client.set('last_symbols_update', str(DateTimeUtils.now()))
        self.redis_client.set('symbols', json.dumps(self.get_symbols()))

    def check_timeouts(self, timestamp):
        now = DateTimeUtils.now()

        timed_out = [symbol[0] for symbol in
                     self.symbol_timestamp.items()
                     if (now - symbol[-1]) > self.timeout]

        for symbol in timed_out:
            self.queue_symbol_restart(symbol)

    def queue_symbol_restart(self, symbol):
        queue = self.symbol_restart_queue

        if symbol not in queue.keys():
            self.add_symbol_restart_queue(symbol)

    def add_symbol_restart_queue(self, symbol):
        last_restart = DateTimeUtils.now() - self.symbol_max_restart_delay
        self.symbol_restart_queue[symbol] = last_restart

    @property
    def next_redis_ip(self):
        max_ix = len(self.redis_network_ips)

        if self.redis_ip_ix > max_ix - 1:
            self.redis_ip_ix = 0

        redis_ip = self.redis_network_ips[self.redis_ip_ix]

        self.redis_ip_ix += 1

        return redis_ip


    @cached_property
    def redis(self):
        return [service for service in self.client.services.list()
                if 'redis' in service.name][0]

    @cached_property
    def redis_network_ips(self):
        return sorted([(network.name, self.redis_ip(network)) for network in
                 self.public_networks])

    @property
    def public_networks(self):
        public_networks = [network for network in self.client.networks.list()
                         if 'public' in network.name]

        return public_networks

    @cached_property
    def network_names(self):
        return [network.name for network in self.public_networks]

    def ip_network_name(self, network):
        return

    def redis_ip(self, network):
        redis = self.redis

        redis_ips = redis.attrs['Endpoint']['VirtualIPs']

        redis_ip = [ip for ip in redis_ips
                   if ip['NetworkID'] == network.id][0]['Addr'].split('/')[0]

        return redis_ip

    def command(self, symbol):
        return f'bash -c "source ~/.bashrc && ' \
               f'./exchange_data/emitters/binance/_depth_emitter.py {symbol}"'

    def get_symbols(self):
        exchange_info = self.bnb_client.get_exchange_info()

        symbols = [symbol for symbol in exchange_info['symbols']
                   if symbol['status'] == 'TRADING']

        symbol_names = [symbol['symbol'] for symbol in symbols if symbol[
            'symbol'].endswith('BTC')]

        # alog.info(alog.pformat(symbol_names))
        #
        # alog.info(len(symbol_names))

        return symbol_names

    def stop_all(self):
        services = self.client.services.list()

        depth_services = [service for service in services
                          if self.is_depth_service(service)]

        for service in depth_services:
            service.remove()

    def exists(self, name):
        services = [service for service in self.client.services.list()
                    if service.name == name]

        return len(services) > 0

    def stop(self, name):
        services = self.client.services.list()

        depth_services = [service for service in services
                          if service.name == name]

        for service in depth_services:
            service.remove()

    def is_depth_service(self, service):
        return re.match(r'binance\_[A-Z]{4,8}\_emit\_depth', service.name)

    def restart(self, symbol):
        redis_ip = self.next_redis_ip
        kwargs = dict(
            name=f'binance_{symbol}_emit_depth',
            image=self.image_name,
            command=self.command(symbol),
            env=['LOG_LEVEL=INFO'],
            networks=[redis_ip[0]],
            hosts={
                'redis': redis_ip[-1]
            }
        )

        alog.info(alog.pformat(kwargs))

        if self.exists(kwargs['name']):
            self.stop(kwargs['name'])

        self.client.services.create(**kwargs)

    def run(self, channels=[]):
        self.sub([
            '2s',
            '30s',
            '1m',
            'symbol_timeout'
        ] + channels)

@click.command()
@click.option('--stop', '-s', is_flag=True)
@click.option('--stop-all', is_flag=True)
@click.option('--restart', '-r', is_flag=True)
def main(**kwargs):
    emitter = BinanceDepthEmitters(**kwargs)

    emitter.run()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda: exit(0))
    signal.signal(signal.SIGTERM, lambda: exit(0))
    main()
