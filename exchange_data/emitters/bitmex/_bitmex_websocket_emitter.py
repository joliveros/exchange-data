from abc import ABC

from bitmex_websocket import Instrument
from bitmex_websocket.constants import InstrumentChannels
from exchange_data import settings
from exchange_data.channels import BitmexChannels
from exchange_data.emitters import Messenger

import alog
import click
import json
import signal
import websocket


class BitmexEmitterBase(Messenger, ABC):
    def __init__(self, symbol: BitmexChannels, **kwargs):
        super().__init__(symbol=symbol, **kwargs)
        self.symbol = symbol


class BitmexEmitter(Instrument, BitmexEmitterBase):
    measurements = []
    channels = [
        # InstrumentChannels.quote,
        InstrumentChannels.trade,
        InstrumentChannels.orderBookL2
    ]

    def __init__(self, symbol: BitmexChannels, **kwargs):
        self.symbol = symbol
        super().__init__(symbol=symbol.name, channels=self.channels, **kwargs)
        # super(Messenger, self).__init__(**kwargs)

        websocket.enableTrace(settings.RUN_ENV == 'development')

        self.on('action', self.on_action)

    def on_action(self, data):
        if not isinstance(data, str):
            data = json.dumps(data)

        msg = self.symbol, data

        self.publish(*msg)

    def start(self):
        self.run_forever()


@click.command()
@click.argument('symbol', type=click.Choice(BitmexChannels.__members__))
def main(symbol: str):
    emitter = BitmexEmitter(BitmexChannels[symbol])
    emitter.start()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda: exit(0))
    signal.signal(signal.SIGTERM, lambda: exit(0))
    main()
