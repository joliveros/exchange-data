#!/usr/bin/env python
import time

import alog
from pytimeparse.timeparse import timeparse
from requests.exceptions import ProxyError, ConnectTimeout, ReadTimeout
from urllib3.exceptions import ConnectTimeoutError

from exchange_data.emitters import Messenger, TimeEmitter

import click
import json
import signal

from exchange_data.emitters.binance import ProxiedClient
from exchange_data.emitters.binance.symbol_emitter import SymbolEmitter


class FullOrderBookEmitter(Messenger):

    def __init__(self, interval: str = '1m', **kwargs):
        super().__init__(**kwargs)
        self.interval = timeparse(interval)

        while True:
            for symbol in self.symbols:
                self.publish_orderbook(symbol)
                time.sleep(self.interval)

    @property
    def client(self):
        return ProxiedClient()

    @property
    def symbols(self):
        return json.loads(SymbolEmitter._symbols())

    def publish_orderbook(self, *args):
        try:
            self._publish_orderbook(*args)
        except (ProxyError, ConnectTimeout, ReadTimeout) as e:
            alog.info(e)
            self.publish_orderbook(*args)

    def _publish_orderbook(self, symbol):
        alog.info(f'## get orderbook for {symbol} ##')
        timestamp = TimeEmitter.timestamp()
        depth = self.client.get_order_book(symbol=symbol, limit=5000)

        data = dict(
            a=depth['asks'],
            b=depth['bids'],
            E=timestamp,
            s=symbol,
        )

        alog.info(len(data['a']) + len(data['b']))

        msg = 'depth_reset', json.dumps(dict(data=data))

        self.publish(*msg)


@click.command()
@click.option('--interval', '-i', type=str, default='1m')
def main(**kwargs):
    emitter = FullOrderBookEmitter(**kwargs)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda: exit(0))
    signal.signal(signal.SIGTERM, lambda: exit(0))
    main()
