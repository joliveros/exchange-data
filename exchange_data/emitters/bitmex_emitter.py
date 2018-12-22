from bitmex_websocket import Instrument
from bitmex_websocket.constants import InstrumentChannels
from exchange_data import settings
from exchange_data.emitters import Messenger
from exchange_data.utils import NoValue

import alog
import json
import sys
import websocket


class BitmexChannels(NoValue):
    XBTUSD = 'XBTUSD-Bitmex'


class BitmexEmitterBase(object):
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.channel = BitmexChannels[symbol.upper()]


class BitmexEmitter(BitmexEmitterBase, Messenger, Instrument):
    measurements = []
    channels = [
        # InstrumentChannels.quote,
        InstrumentChannels.trade,
        InstrumentChannels.orderBookL2
    ]

    def __init__(self, symbol):
        BitmexEmitterBase.__init__(self, symbol)
        Messenger.__init__(self)
        websocket.enableTrace(settings.RUN_ENV == 'development')

        self.symbol = symbol
        Instrument.__init__(self, symbol=symbol,
                            channels=self.channels,
                            should_auth=False)

        self.on('action', self.on_action)

    def on_action(self, data):
        msg = self.channel.value, json.dumps(data)

        try:
            self.publish(*msg)
        except Exception as e:
            alog.info(e)
            sys.exit(-1)

    def start(self):
        self.run_forever()
