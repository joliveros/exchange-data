#!/usr/bin/env python

from collections import deque

import alog
from bitmex import bitmex
from datetime import datetime, timedelta
from exchange_data import Database
from exchange_data._measurement import Measurement
from exchange_data.channels import BitmexChannels
from exchange_data.emitters import SignalInterceptor, TimeChannels, \
    TimeEmitter
from exchange_data.emitters.bitmex import BitmexEmitterBase
from typing import List, Tuple

import click
import json

from exchange_data.utils import DateTimeUtils


class BXBTIndexEmitter(
    BitmexEmitterBase,
    Database,
    SignalInterceptor,
    DateTimeUtils
):

    def __init__(self, interval: str = '1m', **kwargs):
        self.symbol = BitmexChannels.BXBT

        super().__init__(
            symbol=self.symbol,
            database_name='bitmex',
            exit_func=self.stop,
            **kwargs
        )

        self.bitmex_client = bitmex(test=False)

        self.interval = interval
        self.channel = self.generate_channel_name(interval, self.symbol)
        self.prices = deque(maxlen=50)
        self.index_price = None

        self.on(self.interval, self.fetch_price)
        self.on(TimeChannels.Tick.value, self.emit_index)

    @property
    def last_minute(self):
        return datetime.utcnow() - timedelta(minutes=1)

    @staticmethod
    def generate_channel_name(interval: str, symbol: BitmexChannels):
        return f'{symbol.value}_{interval}'

    def fetch_price(self, timestamp, start_time=None):
        if start_time is None:
            start_time = self.last_minute

        indexes = self.bitmex_client.Instrument \
            .Instrument_getCompositeIndex(
                symbol=self.symbol.value,
                startTime=start_time,
                count=3
            ) \
            .result()[0]

        indexes_str, formatted_indexes = self.format_indexes(indexes)

        self.prices += indexes

        msg = self.channel, indexes_str

        self.publish(*msg)
        self.write_points(formatted_indexes)
        self.current_index()

    def format_indexes(self, indexes: List[dict]) -> Tuple[str, List]:

        for index in indexes:
            index['timestamp'] = index['timestamp'].timestamp()
            index['logged'] = index['logged'].timestamp()

        return json.dumps(indexes), indexes

    def write_points(self, data):
        points = [
            Measurement(
                measurement=self.channel,
                time=self.parse_timestamp(index['logged']),
                tags={
                    'symbol': self.symbol.value,
                    'reference': index['reference']
                },
                fields=index
            ).__dict__
            for index in data
        ]

        super().write_points(points, time_precision='ms')

    def emit_index(self, timestamp=None):
        if timestamp is None:
            timestamp = TimeEmitter.timestamp()

        if self.index_price is not None:
            msg = BitmexChannels.BXBT_s.value, self.index_price
            self.publish(*msg)

            measurement = Measurement(
                measurement=BitmexChannels.BXBT_s.value,
                time=self.parse_timestamp(timestamp),
                tags=dict(symbol=BitmexChannels.BXBT.value),
                fields=dict(index=self.index_price))

            super().write_points([measurement.__dict__])

    def current_index(self):
        _index = [item for item in self.prices if item['weight'] is not None]
        timestamps = set([item['timestamp'] for item in _index])
        price_timestamp = [
            (
                timestamp, [index['lastPrice'] for index in _index
                            if index['timestamp'] == timestamp]
            ) for timestamp in timestamps
        ]
        avg_prices = [(pt[0], sum(pt[1]) / float(len(pt[1])))
                      for pt in price_timestamp]
        index_price = avg_prices[0][1]
        self.index_price = index_price

    def start(self):
        self.sub([self.interval, TimeChannels.Tick.value])


@click.command()
@click.option('--interval', '-i', type=str, default='1m')
def main(**kwargs):
    emitter = BXBTIndexEmitter(**kwargs)
    emitter.start()


if __name__ == '__main__':
    main()
