from bitmex_websocket import Instrument
from bitmex_websocket.constants import InstrumentChannels
from exchange_data import settings
from exchange_data.channels import BitmexChannels
from exchange_data.emitters import Messenger

import click
import json
import signal
import websocket


class BitmexEmitterBase(object):
    def __init__(self, symbol: BitmexChannels, **kwargs):
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
