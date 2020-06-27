#!/usr/bin/env python
import alog
from binance.client import Client
from bitmex import bitmex
from exchange_data.channels import BitmexChannels
from exchange_data.emitters import Messenger

import click
import json
import signal


class FullOrderBookEmitter(Messenger):

    def __init__(self, symbol: str, interval: str = '1m', **kwargs):
        super().__init__(**kwargs)
        self.symbol = symbol
        self.bitmex = bitmex(test=False)
        self.interval = interval
        self.channel = self.generate_channel_name(interval, self.symbol)
        self.client = Client()

        self.on(self.interval, self.publish_orderbook)

    @staticmethod
    def generate_channel_name(interval: str, symbol: str):
        return f'{symbol}_OrderBookL2_{interval}'

    def publish_orderbook(self, timestamp):
        depth = self.client.get_order_book(symbol=self.symbol)
        msg = self.channel, json.dumps(depth)
        self.publish(*msg)

    def start(self):
        self.sub([self.interval])


@click.command()
@click.argument('symbol', type=str)
@click.option('--interval', '-i', type=str, default='1m')
def main(**kwargs):
    emitter = FullOrderBookEmitter(**kwargs)
    emitter.start()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda: exit(0))
    signal.signal(signal.SIGTERM, lambda: exit(0))
    main()
