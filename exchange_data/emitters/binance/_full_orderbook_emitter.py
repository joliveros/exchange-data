#!/usr/bin/env python
import concurrent
import time

import alog
from binance.exceptions import BinanceAPIException
from pytimeparse.timeparse import timeparse
from requests.exceptions import ProxyError, ConnectTimeout, ReadTimeout, \
    SSLError
from urllib3.exceptions import ConnectTimeoutError

from exchange_data.emitters.binance import BinanceUtils
from exchange_data.emitters import Messenger, TimeEmitter

import click
import json
import signal

from exchange_data.emitters.binance import ProxiedClient
from exchange_data.emitters.binance.symbol_emitter import SymbolEmitter


class FullOrderBookEmitter(Messenger, BinanceUtils):

    def __init__(self, symbol, futures, depth, **kwargs):
        super().__init__(**kwargs)
        self.futures = futures
        self.depth = depth
        self.publish_orderbook(symbol)

    @property
    def client(self):
        return ProxiedClient()

    def publish_orderbook(self, *args):
        try:
            self._publish_orderbook(*args)
        except (ProxyError, ConnectTimeout, ReadTimeout, SSLError, BinanceAPIException) as e:
            alog.info(e)
            self.publish_orderbook(*args)

    def _publish_orderbook(self, symbol):
        alog.info(f'## get orderbook for {symbol} ##')
        timestamp = TimeEmitter.timestamp()

        if self.futures:
            depth = self.client.futures_order_book(symbol=symbol, limit=self.depth)
        else:
            depth = self.client.get_order_book(symbol=symbol, limit=self.depth)

        alog.info(depth)

        data = dict(
            a=depth['asks'],
            b=depth['bids'],
            E=timestamp,
            s=symbol,
        )

        alog.info(len(data['a']) + len(data['b']))

        if self.futures:
            channel = f'{symbol}_depth_reset_futures'
        else:
            channel = f'{symbol}_depth_reset'

        msg = channel, json.dumps(dict(data=data))
        alog.info(msg)

        self.publish(*msg)


@click.command()
@click.option('--futures', '-F', is_flag=True)
@click.option('--depth', '-d', default=5000, type=int)
def main(futures, **kwargs):
    def emit_orderbook(symbol):
        FullOrderBookEmitter(symbol, futures, **kwargs)

    symbols = BinanceUtils(futures=futures, symbol_filter=None).symbols

    with concurrent.futures.ThreadPoolExecutor() as executor:
        while len(symbols) > 0:
            symbol = symbols.pop()
            executor.submit(emit_orderbook, symbol)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda: exit(0))
    signal.signal(signal.SIGTERM, lambda: exit(0))
    main()
