from bitmex_websocket.constants import InstrumentChannels

from .. import settings
from . import Recorder
from bitmex_websocket import Instrument
import alog
import websocket

alog.set_level(settings.LOG_LEVEL)


class BitmexRecorder(Recorder, Instrument):
    measurements = []
    channels = [
        InstrumentChannels.quote,
        InstrumentChannels.trade,
        InstrumentChannels.orderBookL2
    ]

    def __init__(self, symbol, database_name='bitmex'):
        super().__init__(symbol, database_name)
        websocket.enableTrace(settings.RUN_ENV == 'development')

        self.symbol = symbol
        Instrument.__init__(self, symbol=symbol,
                            channels=self.channels,
                            should_auth=False)

        self.on('action', self.on_action)

    def on_action(self, data):
        data['symbol'] = self.symbol
        data['timestamp'] = self.get_timestamp()

        for row in data['data']:
            row.pop('symbol', None)

        alog.debug(alog.pformat(data))
        self.save_measurement('data', self.symbol, data)

    def start(self):
        self.run_forever()

