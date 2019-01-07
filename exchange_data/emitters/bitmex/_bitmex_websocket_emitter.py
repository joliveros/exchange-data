from bitmex_websocket import Instrument
from bitmex_websocket.constants import InstrumentChannels, NoValue
from exchange_data import settings
from exchange_data.emitters import Messenger

import alog
import json
import sys
import websocket


class BitmexChannels(NoValue):
    XBTUSD = 'XBTUSD'


class BitmexEmitterBase(object):
    def __init__(self, symbol: BitmexChannels):
        self.symbol = symbol


class BitmexEmitter(BitmexEmitterBase, Messenger, Instrument):
    measurements = []
    channels = [
        # InstrumentChannels.quote,
        InstrumentChannels.trade,
        InstrumentChannels.orderBookL2
    ]

    def __init__(self, symbol: BitmexChannels):
        BitmexEmitterBase.__init__(self, symbol)
        Messenger.__init__(self)
        websocket.enableTrace(settings.RUN_ENV == 'development')

        Instrument.__init__(self, symbol=symbol.value,
                            channels=self.channels,
                            should_auth=False)

        self.on('action', self.on_action)

    def on_action(self, data):
        msg = self.symbol, json.dumps(data)

        self.publish(*msg)

    def start(self):
        self.run_forever()
