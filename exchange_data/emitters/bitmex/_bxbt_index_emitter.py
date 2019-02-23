from bitmex import bitmex
from datetime import datetime, timedelta
from exchange_data import Database
from exchange_data._measurement import Measurement
from exchange_data.channels import BitmexChannels
from exchange_data.emitters import Messenger
from exchange_data.emitters.bitmex import BitmexEmitterBase
from pyee import EventEmitter
from typing import List, Callable

import click
import json
import signal


class SignalInterceptor(object):
    def __init__(self, exit_func: Callable):
        signal.signal(signal.SIGINT, exit_func)
        signal.signal(signal.SIGTERM, exit_func)


class BXBTIndexEmitter(
    BitmexEmitterBase,
    Messenger,
    Database,
    SignalInterceptor
):

    def __init__(self, interval: str = '1m', **kwargs):
        self.symbol = BitmexChannels.BXBT

        BitmexEmitterBase.__init__(self, symbol=self.symbol, **kwargs)
        Database.__init__(self, database_name='bitmex', **kwargs)
        Messenger.__init__(self)
        SignalInterceptor.__init__(self, self.stop)

        self.bitmex_client = bitmex()

        self.interval = interval
        self.channel = self.generate_channel_name(interval, self.symbol)

        self.on(self.interval, self.fetch_price)

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

        msg = self.channel, indexes_str
        self.publish(*msg)

        self.write_points(formatted_indexes)

    def format_indexes(self, indexes: List[dict]) -> (str, List):
        formatted_indexes = []

        for index in indexes:
            index['timestamp'] = index['timestamp'].timestamp() * 1000
            index['logged'] = index['logged'].timestamp() * 1000
            formatted_indexes.append(index)

        return json.dumps(formatted_indexes), formatted_indexes

    def write_points(self, data):
        points = [
            Measurement(
                measurement=self.channel,
                timestamp=index['timestamp'],
                tags={'symbol': self.symbol.value},
                fields=index
            ).__dict__
            for index in data
        ]

        super().write_points(points, time_precision='ms')

    def start(self):
        self.sub([self.interval])


@click.command()
@click.option('--interval', '-i', type=str, default='1m')
@click.option('--influxdb', type=str)
def main(**kwargs):
    emitter = BXBTIndexEmitter(**kwargs)
    emitter.start()


if __name__ == '__main__':
    main()
