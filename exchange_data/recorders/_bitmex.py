from bitmex_websocket.constants import InstrumentChannels
from exchange_data import settings
from exchange_data.channels import BitmexChannels
from exchange_data.emitters import Messenger, TimeEmitter
from exchange_data.emitters.bitmex._orderbook_l2_emitter import OrderBookL2Emitter
from exchange_data.recorders import Recorder

import alog
import click
import json
import signal
import sys

from exchange_data.utils import DateTimeUtils

alog.set_level(settings.LOG_LEVEL)


class BitmexRecorder(Recorder, Messenger, DateTimeUtils):
    measurements = []
    channels = [
        InstrumentChannels.quote,
        InstrumentChannels.trade,
        InstrumentChannels.orderBookL2
    ]

    def __init__(
        self,
        measurement_name,
        symbol: BitmexChannels,
        database_name='bitmex',
        no_save: bool = False,
        **kwargs,
    ):
        self.measurement_name = measurement_name
        if isinstance(symbol, str):
            symbol = BitmexChannels[symbol]

        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        DateTimeUtils.__init__(self)
        Messenger.__init__(self)
        Recorder.__init__(self, symbol, database_name, **kwargs)

        self.symbol = symbol
        self.no_save = no_save
        self.orderbook_l2_channel = \
            OrderBookL2Emitter.generate_channel_name('1m', self.symbol)

        self.on(self.symbol.value, self.on_action)
        self.on(self.orderbook_l2_channel, self.on_action)

    def on_action(self, data):
        if isinstance(data, str):
            data = json.loads(data)

        # TODO: let the orderbookL2 emitter produce this correctly.
        if isinstance(data, list):
            data = {
                'table': 'orderBookL2',
                'data': data,
                'action': 'partial',
                'symbol': self.symbol.value
            }

        data['symbol'] = self.symbol.value

        if 'timestamp' in data:
            data['time'] = self.parse_timestamp(data['timestamp'])
        else:
            data['time'] = self.now()

        for row in data['data']:
            row.pop('symbol', None)

        if not self.no_save:
            self.save_measurement(self.measurement_name,
                                  self.symbol.value, data)

    def stop(self, *args):
        self.close()
        sys.exit(0)

    def start(self):
        self.sub([
            self.symbol,
            self.orderbook_l2_channel
        ])


@click.command()
@click.argument('symbol', type=click.Choice(BitmexChannels.__members__))
@click.option('--no-save', '-n', is_flag=True, help='disable saving to disk')
@click.option('--batch-size', '-b', type=int, default=100, help='batch size at which to save to influxdb.')
@click.option('--influxdb', type=str, default=None, help='override influxdb connection string.')
@click.option('--measurement-name', type=str, default=None, help='Measurent name.')
def main(symbol: str, **kwargs):
    emitter = BitmexRecorder(symbol=BitmexChannels[symbol], **kwargs)
    emitter.start()


if __name__ == '__main__':
    main()
