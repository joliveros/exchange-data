from exchange_data import Database
from exchange_data import settings
from datetime import datetime

import alog
import json

from exchange_data.emitters import TimeEmitter

alog.set_level(settings.LOG_LEVEL)


class Recorder(Database):
    measurements = []
    channels = []
    symbols = []

    def __init__(self, symbols, database_name, batch_size: int = 100, **kwargs):
        Database.__init__(self, database_name=database_name, **kwargs)

        self.batch_size = batch_size
        self.symbols = symbols

    def to_lowercase_keys(self, data):
        return dict((k.lower(), v) for k, v in data.items())

    def values_to_str(self, data):
        keys = data.keys()

        for key in keys:
            value = data[key]
            if isinstance(value, (dict, list)):
                data[key] = json.dumps(value)

        return data

    def save_measurement(self, name, symbol, table):
        if isinstance(table, str):
            table = json.loads(table)

        measurement = {
            'measurement': name,
            'tags': {
                'symbol': symbol
            },
            'timestamp': self.get_timestamp(table),
            'fields': {
                'data': json.dumps(table)
            }
        }

        self.measurements.append(measurement)

        if len(self.measurements) >= self.batch_size:
            self.write_points(self.measurements, time_precision='ms')

            self.measurements = []

    def get_timestamp(self, table):
        return float(
            table.get('microtimestamp', TimeEmitter.timestamp() * 1000)) / 1000

