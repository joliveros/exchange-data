#!/usr/bin/env python
import enum

from bitmex import bitmex

from exchange_data.channels import BitmexChannels
from exchange_data.emitters import Messenger

import click
import json
import signal


class OrderBookL2Emitter(Messenger):

    def __init__(self, symbol: BitmexChannels, interval: str = '1m', **kwargs):
        super().__init__(**kwargs)
        self.symbol = symbol
        self.bitmex = bitmex(test=False)
        self.interval = interval
        self.channel = self.generate_channel_name(interval, self.symbol)

        self.on(self.interval, self.publish_orderbook)

    @staticmethod
    def generate_channel_name(interval: str, symbol: BitmexChannels):
        _symbol = symbol
        if issubclass(type(symbol), enum.Enum):
            _symbol = symbol.value

        return f'{symbol}_OrderBookL2_{interval}'


    def publish_orderbook(self, timestamp):
        data = self.bitmex.OrderBook.OrderBook_getL2(
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
