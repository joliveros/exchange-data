#!/usr/bin/env python

from exchange_data.bitmex_orderbook import BitmexOrderBook
from exchange_data.emitters.bitmex import BitmexEmitterBase
from exchange_data.emitters import Messenger, TimeChannels

import click
import json
import signal

signal.signal(signal.SIGINT, lambda: exit(0))
signal.signal(signal.SIGTERM, lambda: exit(0))


class LiveBitmexOrderBook(BitmexEmitterBase, BitmexOrderBook, Messenger):

    def __init__(self, symbol: str):
        BitmexEmitterBase.__init__(self, symbol)
        Messenger.__init__(self)
        BitmexOrderBook.__init__(self, symbol=symbol)

    def _message(self, msg):
        if msg.get('type') != 'subscribe':
            data = json.loads(msg['data'])

            if data['table'] == 'orderBookL2':
                self.orderbook.emit('orderBookL2', data)

    def start(self):
        channels = [self.channel, TimeChannels.Tick]
        super().sub(channels)


@click.command()
def main():
    orderbook: LiveBitmexOrderBook = LiveBitmexOrderBook('XBTUSD')

    orderbook.start()
