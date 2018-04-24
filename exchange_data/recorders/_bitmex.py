from .. import settings
from . import Recorder
from bitmex_websocket import Instrument
import alog
import websocket

alog.set_level(settings.LOG_LEVEL)


class BitmexRecorder(Recorder):
    measurements = []
    channels = [
        'quote',
        'trade',
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

        instrument.on('latency',
                      lambda latency: alog.debug(
                          "latency: {0}".format(latency)))
        instrument.on('action', self.on_action)

    def on_action(self, data):
        data = self.to_lowercase_keys(data)
        data['symbol'] = data['data'][0]['symbol']
        data['timestamp'] = self.get_timestamp()

        for row in data['data']:
            row.pop('symbol', None)

        self.save_measurement('data', data['symbol'], data)
