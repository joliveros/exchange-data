import json

from bitmex import bitmex
from bravado.client import SwaggerClient
from pytimeparse.timeparse import timeparse

from exchange_data import settings
from exchange_data.channels import BitmexChannels
from exchange_data.emitters import Messenger, TimeChannels
from exchange_data.emitters.bitmex import BitmexEmitterBase

import alog
import click
import signal


class OrderBookL2Emitter(BitmexEmitterBase, Messenger, SwaggerClient):

    def __init__(self, symbol: BitmexChannels, interval: str = '1m'):
        BitmexEmitterBase.__init__(self, symbol)
        Messenger.__init__(self)
        SwaggerClient.__init__(self, None)
        bitmex_client = bitmex()

        self.interval = interval
        self.channel = self.generate_channel_name(interval, self.symbol)

        self.__dict__ = {
            **self.__dict__,
            **bitmex_client.__dict__
        }

        self.on(self.interval, self.publish_orderbook)

    @staticmethod
    def generate_channel_name(interval: str, symbol: BitmexChannels):
        return f'{symbol.value}_OrderBookL2_{interval}'

    def publish_orderbook(self, timestamp):
        data = self.OrderBook.OrderBook_getL2(
            symbol=self.symbol.value,
            depth=0
        ).result()

        msg = self.channel, json.dumps(data[0])
        self.publish(*msg)

    def start(self):
        self.sub([self.interval])


@click.command()
@click.argument('symbol', type=click.Choice(BitmexChannels.__members__))
@click.option('--interval', '-i', type=str, default='1m')
def main(symbol: str, interval: str):
    emitter = OrderBookL2Emitter(BitmexChannels[symbol], interval)
    emitter.start()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda: exit(0))
    signal.signal(signal.SIGTERM, lambda: exit(0))
    main()
