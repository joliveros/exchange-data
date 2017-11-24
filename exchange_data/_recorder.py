from . import Database
from . import settings
from datetime import datetime
import alog
import json

alog.set_level(settings.LOG_LEVEL)


class Recorder(Database):
    measurements = []
    channels = []
    symbols = []

    def __init__(self, symbols, database_name):
        Database.__init__(self, database_name=database_name)

        self.symbols = symbols

    def pp(self, data):
        if settings.LOG_LEVEL == 'DEBUG':
            alog.debug(json.dumps(data, indent=2, sort_keys=True))

    def to_lowercase_keys(self, data):
        return dict((k.lower(), v) for k, v in data.items())

    def values_to_str(self, data):
        keys = data.keys()

        for key in keys:
            value = data[key]
            if isinstance(value, (dict, list)):
                data[key] = json.dumps(value)

        return data

    def get_timestamp(self):
        return f'{str(datetime.utcnow())}Z'

    def save_measurement(self, name, symbol, table):
        if isinstance(table, str):
            table = json.loads(table)

        measurement = {
            'measurement': name,
            'tags': {
                'symbol': symbol
            },
            'timestamp': table.get('timestamp'),
            'fields': {
                'data': json.dumps(table)
            }
        }
        self.pp(measurement)
        self.measurements.append(measurement)

        if len(self.measurements) >= settings.MEASUREMENT_BATCH_SIZE:
            alog.debug('### Save measurements count: %s' % len(self.measurements))
            # self.pp(self.measurements)
            self.write_points(self.measurements, time_precision='ms')

            self.measurements = []
