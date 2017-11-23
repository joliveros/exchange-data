from . import settings
from . import Database
from bitmex_websocket import Instrument
from datetime import datetime
import alog
import json
import websocket

alog.set_level(settings.LOG_LEVEL)


class BitmexRecorder(Database):
    measurements = []
    channels = [
        'trade',
        'quote',
        'orderBookL2'
    ]

    def __init__(self, symbol):
        super().__init__(database_name='bitmex')
        websocket.enableTrace(settings.RUN_ENV == 'development')

        self.symbols = symbol

        for symbol in self.symbols:
            self.subscribe_symbol(symbol)

    def subscribe_symbol(self, symbol):
        instrument = Instrument(symbol=symbol,
                                channels=self.channels,
                                # set to 1 because data will be saved to db
                                shouldAuth=False)

        for table in self.channels:
            instrument.on(table, self.on_table)

        instrument.on('latency',
                      lambda latency: print("latency: {0}".format(latency)))
        instrument.on('action', self.on_action)

    def on_table(self, table_name, table):
        if table_name == 'trade':
            return self.on_trade(table)
        elif table_name == 'quote':
            return self.on_quote(table)

    def on_action(self, message):
        table = message['table']

        if table == 'orderBookL2':
            data = message.copy()
            data = self.to_lowercase_keys(data)
            data['symbol'] = data['data'][0]['symbol']
            data['timestamp'] = self.get_timestamp()
            data.pop('table', None)

            for row in data['data']:
                row.pop('symbol', None)

            data = self.values_to_str(data)
            self.save_measurement(table, data)

    def pp(self, data):
        if settings.LOG_LEVEL == 'DEBUG':
            alog.debug(json.dumps(data, indent=2, sort_keys=True))

    def get_timestamp(self):
        return f'{str(datetime.utcnow())}Z'

    def on_quote(self, table):
        data = table.copy()
        data = self.to_lowercase_keys(data)
        data['bidsize'] = str(data['bidsize'])
        data['bidprice'] = str(data['bidprice'])
        data['asksize'] = str(data['asksize'])
        data['askprice'] = str(data['askprice'])
        data = self.values_to_str(data)
        self.save_measurement('quote', data)

    def on_trade(self, table):
        data = table.copy()
        data = self.to_lowercase_keys(data)
        data.pop('homenotional', None)
        data.pop('foreignnotional', None)
        data.pop('grossvalue', None)
        data['price'] = float(data['price'])
        data = self.values_to_str(data)
        self.save_measurement('trade', data)

    def to_lowercase_keys(self, data):
        return dict((k.lower(), v) for k, v in data.items())

    def values_to_str(self, data):
        keys = data.keys()

        for key in keys:
            value = data[key]
            if isinstance(value, (dict, list)):
                data[key] = json.dumps(value)

        return data

    def save_measurement(self, name, table):
        timestamp = table.pop('timestamp', None)
        symbol = table.pop('symbol', None)

        measurement = {
            'measurement': name,
            'tags': {
                'symbol': symbol
            },
            'time': timestamp,
            'fields': table
        }

        self.measurements.append(measurement)

        if len(self.measurements) >= settings.MEASUREMENT_BATCH_SIZE:
            alog.debug('### Save measurements count: %s' % len(self.measurements))
            self.pp(self.measurements)
            self.write_points(self.measurements, time_precision='ms')

            self.measurements = []
