from . import settings
from . import Database
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

    def save_measurement(self, name, symbol, table):
        table = json.loads(table)
        timestamp = table.pop('timestamp', None)

        measurement = {
            'measurement': name,
            'tags': {
                'symbol': symbol
            },
            # 'time': timestamp,
            'fields': {
                'data': json.dumps(table)
            }
        }
        # self.pp(measurement)
        self.measurements.append(measurement)

        if len(self.measurements) >= settings.MEASUREMENT_BATCH_SIZE:
            alog.debug('### Save measurements count: %s' % len(self.measurements))
            # self.pp(self.measurements)
            try:
                self.write_points(self.measurements, time_precision='ms')
            except Exception as e:
                alog.error(e)
                raise e

            self.measurements = []
