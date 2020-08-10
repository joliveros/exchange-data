#!/usr/bin/env python


from dateutil import tz
from exchange_data import settings, Database, Measurement
from exchange_data.emitters import Messenger, SignalInterceptor
from exchange_data.utils import DateTimeUtils
from functools import lru_cache

import alog
import click
import json
import sys
import traceback
import numpy as np


class TradeRecorder(
    Messenger,
    Database,
    SignalInterceptor,
    DateTimeUtils,
):
    def __init__(
        self,
        database_name='binance',
        subscriptions_enabled: bool = True,
        **kwargs
    ):
        self.subscriptions_enabled = subscriptions_enabled

        super().__init__(
            database_name=database_name,
            **kwargs
        )

        self.last_frames = {}
        self.slices = {}
        self.queued_frames = []
        self.last_timestamp = DateTimeUtils.now()

        self.freq = settings.TICK_INTERVAL

        if self.subscriptions_enabled:
            self.on('trade', self.message)

    def message(self, data):
        timestamp = data['timestamp']
        del data['timestamp']
        symbol = data['symbol']
        del data['symbol']

        m = Measurement(measurement='trade',
                    fields=data,
                    tags=dict(symbol=symbol),
                    time=timestamp)

        self.write_points([vars(m)], time_precision='ms')

    def start(self, channels=[]):
        self.sub([
            'trade',
        ] + channels)

    def exit(self, *args):
        self.stop()
        sys.exit(0)


@click.command()
@click.option('--database-batch-size', '-b', type=int, default=300)
def main(**kwargs):
    recorder = TradeRecorder(
        subscriptions_enabled=True,
        **kwargs
    )

    try:
        recorder.start()
    except Exception as e:
        traceback.print_exc()
        recorder.stop()
        sys.exit(-1)


if __name__ == '__main__':
    main()
