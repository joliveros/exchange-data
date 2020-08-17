#!/usr/bin/env python
from binance.client import Client

from exchange_data import Database, settings
from exchange_data.emitters import Messenger
from exchange_data.utils import DateTimeUtils

import alog
import click
import signal
import sys


class BinanceAccountEmitter(
    Database,
    Messenger,
    DateTimeUtils,
):
    measurements = []

    def __init__(self, **kwargs):
        super().__init__(
            **kwargs
        )

        self.client = Client(
            api_key=settings.BINANCE_API_KEY,
            api_secret=settings.BINANCE_API_SECRET
        )

    def start(self):

    def stop(self):
        sys.exit(0)


@click.command()
@click.option('--database-name', '-d', default='binance', type=str)
def main(**kwargs):
    emitter = BinanceAccountEmitter(**kwargs)
    emitter.start()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda: exit(0))
    signal.signal(signal.SIGTERM, lambda: exit(0))
    main()
