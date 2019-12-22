import logging

from dateutil.tz import tz

from exchange_data import Database
from exchange_data import settings
from datetime import datetime

import alog
import json

from exchange_data.emitters import TimeEmitter
from exchange_data.utils import DateTimeUtils

alog.set_level(settings.LOG_LEVEL)


class Recorder(Database, DateTimeUtils):
    measurements = []
    channels = []

    def __init__(self, symbol, database_name, batch_size: int = 100, **kwargs):
        super().__init__(database_name=database_name, **kwargs)
        self.batch_size = batch_size
        self.symbol = symbol

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

        time = table.get('timestamp', None)

        if time is None:
            time = table.get('time', self.now())

        if isinstance(time, str):
            time = self.parse_timestamp(time)

        if 'time' in table:
            del table['time']

        if 'timestamp' in table:
            del table['timestamp']

        measurement = {
            'measurement': name,
            'tags': {
                'symbol': symbol
            },
            'time': time,
            'fields': {
                'data': json.dumps(table)
            }
        }

        if settings.LOG_LEVEL == logging.DEBUG:
            alog.info(alog.pformat(measurement))

        self.measurements.append(measurement)

        if len(self.measurements) >= self.batch_size:
            self.write_points(self.measurements, time_precision='ms')

            self.measurements = []

    def get_timestamp(self, table):
        return float(
            table.get('microtimestamp', TimeEmitter.timestamp() * 1000)) / 1000

