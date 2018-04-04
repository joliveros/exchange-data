from .. import settings
from . import Recorder
from bitmex_websocket import Instrument
import alog
import websocket

alog.set_level(settings.LOG_LEVEL)


class BitmexRecorder(Recorder):
    measurements = []
    channels = [
        'trade',
        'quote',
        'orderBookL2'
    ]

    def __init__(self, symbols):
        super().__init__(symbols, database_name='bitmex')
        websocket.enableTrace(settings.RUN_ENV == 'development')

        self.symbols = symbols

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
                      lambda latency: alog.debug("latency: {0}".format(latency)))
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

            self.save_measurement('orderbook', data['symbol'], data)

    def on_quote(self, table):
        self.save_measurement('quote', table['symbol'], table)

    def on_trade(self, table):
        self.save_measurement('trade', table['symbol'], table)
