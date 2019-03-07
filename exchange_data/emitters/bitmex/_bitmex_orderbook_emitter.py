from collections import deque
from exchange_data import settings, Database, Measurement
from exchange_data.bitmex_orderbook import BitmexOrderBook
from exchange_data.channels import BitmexChannels
from exchange_data.emitters import Messenger, TimeChannels, SignalInterceptor
from exchange_data.emitters.bitmex import BitmexEmitterBase
from exchange_data.emitters.bitmex._orderbook_l2_emitter import \
    OrderBookL2Emitter
from exchange_data.orderbook._ordertree import OrderTree
from exchange_data.utils import NoValue
from numpy.core.multiarray import ndarray

import alog
import click
import gc
import json
import numpy as np
import sys
import traceback


class BitmexOrderBookChannels(NoValue):
    OrderBookFrame = 'OrderBookFrame'


class BitmexOrderBookEmitter(
    BitmexEmitterBase,
    Messenger,
    BitmexOrderBook,
    Database,
    SignalInterceptor
):
    def __init__(
        self,
        symbol: BitmexChannels,
        save_data: bool = True,
        reset_orderbook: bool = True,
        **kwargs
    ):
        BitmexOrderBook.__init__(self, symbol)
        Messenger.__init__(self)
        BitmexEmitterBase.__init__(self, symbol)
        Database.__init__(self, database_name='bitmex')
        SignalInterceptor.__init__(self, self.exit)

        self.save_data = save_data
        self.orderbook_l2_channel = \
            OrderBookL2Emitter.generate_channel_name('1m', self.symbol)
        self.freq = settings.TICK_INTERVAL
        self.frame_channel = f'{self.symbol.value}_' \
            f'{BitmexOrderBookChannels.OrderBookFrame.value}'
        self.on(TimeChannels.Tick.value, self.save_frame)
        self.on(self.symbol.value, self.message)

        if reset_orderbook:
            self.on(self.orderbook_l2_channel, self.process_orderbook_l2)

    def print_stats(self):
        alog.info(self.print(depth=4))
        alog.info(self.dataset.dims)

    def process_orderbook_l2(self, data):
        self.reset_orderbook()

        self.message({
            'table': 'orderBookL2',
            'data': data,
            'action': 'partial',
            'symbol': self.symbol.value
        })

    def reset_orderbook(self):
        alog.info('### reset orderbook ###')
        del self.__dict__['tape']
        del self.__dict__['bids']
        del self.__dict__['asks']

        self.tape = deque(maxlen=10)
        self.bids = OrderTree()
        self.asks = OrderTree()

    def garbage_collect(self):
        gc.collect()

    def start(self):
        self.sub([
            self.symbol,
            TimeChannels.Tick,
            self.orderbook_l2_channel
        ])

    def exit(self, *args):
        self.stop()
        sys.exit(0)

    def gen_bid_side(self):
        bid_levels = list(self.bids.price_tree.items())
        price = np.array(
            [level[0] for level in bid_levels]
        )

        volume = np.array(
            [level[1].volume for level in bid_levels]
        ) * -1

        return np.vstack((price, volume))

    def gen_ask_side(self):
        bid_levels = list(self.asks.price_tree.items())
        price = np.array(
            [level[0] for level in bid_levels]
        )

        volume = np.array(
            [level[1].volume for level in bid_levels]
        )

        return np.vstack((price, volume))

    def generate_frame(self) -> ndarray:
        bid_side = np.flip(self.gen_bid_side(), axis=1)
        ask_side = self.gen_ask_side()

        bid_side_shape = bid_side.shape
        ask_side_shape = ask_side.shape

        if bid_side_shape[-1] > ask_side_shape[-1]:
            new_ask_side = np.zeros(bid_side_shape)
            new_ask_side[:ask_side_shape[0], :ask_side_shape[1]] = ask_side
            ask_side = new_ask_side
        elif ask_side_shape[-1] > bid_side_shape[-1]:
            new_bid_side = np.zeros(ask_side_shape)
            new_bid_side[:bid_side_shape[0], :bid_side_shape[1]] = bid_side
            bid_side = new_bid_side

        frame = np.array((ask_side, bid_side))

        return frame

    def save_frame(self, timestamp):
        self.last_timestamp = timestamp

        frame = self.generate_frame()

        alog.info('\n' + str(frame[:, :, :5]))
        alog.info(frame.shape)

        measurement = Measurement(
            measurement=self.frame_channel,
            tags={'symbol': self.symbol.value},
            timestamp=self.last_timestamp,
            fields={'data': json.dumps(frame.tolist())}
        )

        if self.save_data:
            self.write_points([measurement.__dict__], time_precision='ms')


@click.command()
@click.argument('symbol', type=click.Choice(BitmexChannels.__members__))
@click.option('--save-data/--no-save-data', default=True)
@click.option('--reset-orderbook/--no-reset-orderbook', default=True)
def main(symbol: str, **kwargs):
    recorder = BitmexOrderBookEmitter(
        symbol=BitmexChannels[symbol],
        **kwargs
    )

    try:
        recorder.start()
    except Exception as e:
        traceback.print_exc()
        recorder.stop()
        sys.exit(-1)


if __name__ == '__main__':
    main()
