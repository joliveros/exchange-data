#!/usr/bin/env python

from binance.client import Client
from exchange_data import Database, settings
from exchange_data.data.measurement_frame import MeasurementFrame

import alog
import click
import signal
import sys

from exchange_data.emitters import Messenger


class BinanceAccountEmitter(MeasurementFrame, Messenger):
    measurements = []

    def __init__(self, **kwargs):
        super().__init__(
            **kwargs
        )

        self.client = Client(
            api_key=settings.BINANCE_API_KEY,
            api_secret=settings.BINANCE_API_SECRET
        )

        data = dict(
            account_status=self.client.get_account_status(),
            info=self.client.get_account(),
        )

        alog.info(alog.pformat(data))

    def start(self):
        self.sub(['30s'])

    def stop(self):
        sys.exit(0)


@click.command()
@click.option('--database-name', '-d', default='binance', type=str)
def main(**kwargs):
    emitter = BinanceAccountEmitter(**kwargs)
    # emitter.start()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda: exit(0))
    signal.signal(signal.SIGTERM, lambda: exit(0))
    main()
