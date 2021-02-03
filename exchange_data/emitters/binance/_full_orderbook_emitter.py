#!/usr/bin/env python
import concurrent
import time

import alog
from pytimeparse.timeparse import timeparse
from requests.exceptions import ProxyError, ConnectTimeout, ReadTimeout, \
    SSLError
from urllib3.exceptions import ConnectTimeoutError

from exchange_data.emitters import Messenger, TimeEmitter

import click
import json
import signal

from exchange_data.emitters.binance import ProxiedClient
from exchange_data.emitters.binance.symbol_emitter import SymbolEmitter


class FullOrderBookEmitter(Messenger):

    def __init__(self, symbol, **kwargs):
        super().__init__(**kwargs)

        self.publish_orderbook(symbol)

    @property
    def client(self):
        return ProxiedClient()

    @property
    def symbols(self):
        return json.loads(SymbolEmitter._symbols())

    def publish_orderbook(self, *args):
        try:
            self._publish_orderbook(*args)
        except (ProxyError, ConnectTimeout, ReadTimeout, SSLError) as e:
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

        msg = f'{symbol}_depth_reset', json.dumps(dict(data=data))
        alog.info(msg)

        self.publish(*msg)


@click.command()
def main(**kwargs):
    def emit_orderbook(symbol):
        FullOrderBookEmitter(symbol, **kwargs)

    symbols = json.loads(SymbolEmitter._symbols())

    with concurrent.futures.ThreadPoolExecutor() as executor:
        while len(symbols) > 0:
            symbol = symbols.pop()
            executor.submit(emit_orderbook, symbol)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda: exit(0))
    signal.signal(signal.SIGTERM, lambda: exit(0))
    main()
