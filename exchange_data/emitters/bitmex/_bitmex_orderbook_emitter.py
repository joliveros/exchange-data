import alog

from exchange_data import settings
from exchange_data.bitmex_orderbook import BitmexOrderBook
from exchange_data.channels import BitmexChannels
from exchange_data.emitters import Messenger, TimeChannels
from exchange_data.emitters.bitmex import BitmexEmitterBase
from exchange_data.emitters.websocket_emitter import WebsocketEmitter
from exchange_data.utils import NoValue, MemoryTracing
from exchange_data.xarray_recorders.bitmex import RecorderAppend
from numpy.core.multiarray import ndarray
from pandas import date_range
from xarray import Dataset

import click
import json
import numpy as np
import signal
import sys
import xarray as xr


class BitmexOrderBookChannels(NoValue):
    OrderBookFrame = 'OrderBookFrame'


class BitmexOrderBookEmitter(
    BitmexEmitterBase,
    Messenger,
    BitmexOrderBook,
    RecorderAppend,
    WebsocketEmitter,
    MemoryTracing
):
    def __init__(
            self,
            symbol: BitmexChannels,
            **kwargs
    ):
        BitmexOrderBook.__init__(self, symbol)
        Messenger.__init__(self)
        BitmexEmitterBase.__init__(self, symbol)
        RecorderAppend.__init__(self, symbol=symbol, **kwargs)
        WebsocketEmitter.__init__(self)
        MemoryTracing.__init__(self, **kwargs)

        self.freq = settings.TICK_INTERVAL
        self.frame_channel = f'{self.symbol.value}_' \
            f'{BitmexOrderBookChannels.OrderBookFrame.value}'
        self.on(TimeChannels.Tick.value, self.update_dataset)
        self.on(self.symbol.value, self.message)
        self.on('save', self.trace_print)

    def start(self):
        self.sub([self.symbol, TimeChannels.Tick])

    def stop(self):
        self._pubsub.close()
        self.client.disconnect()
        self.stopped = True
        self.to_netcdf()

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
        prev_frame_shape = (0, 0)
        dataset = self.dataset
        bid_side = self.gen_bid_side()
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

        if dataset:
            prev_frame_shape = \
                dataset.sel(time=dataset.time[-1]).orderbook.shape

        if frame.shape[-1] > prev_frame_shape[-1]:
            if dataset:
                self.resize_dataset(frame)

        elif frame.shape[-1] < prev_frame_shape[-1]:
                return self.resize_frame(frame)

        return frame

    def resize_frame(self, frame):
        book_shape = self.dataset.orderbook.values.shape
        shape = (book_shape[1], book_shape[2], book_shape[3])
        resized_values = np.zeros(shape)
        frame_shape = frame.shape
        resized_values[:frame_shape[0], :frame_shape[1], :frame_shape[2]] = frame
        return resized_values

    def resize_dataset(self, frame):
        book_shape = self.dataset.orderbook.values.shape
        resized_values = np.zeros((book_shape[0],) + frame.shape)

        resized_values[
            :book_shape[0],
            :book_shape[1],
            :book_shape[2],
            :book_shape[3]
        ] = self.dataset.orderbook.values

        self.dataset = self.dataset_frame(
            resized_values,
            self.dataset.time
        )

    def dataset_frame(self, values, time):
        return Dataset(
            {
                'orderbook': (['time', 'frame', 'side', 'levels'], values)
            },
            coords={'time': time}
        )

    def update_dataset(self, timestamp) -> Dataset:
        self.last_timestamp = timestamp

        frame = self.generate_frame()

        time_index = date_range(
            start=self.last_date,
            end=self.last_date,
            freq=self.freq
        ).floor(self.freq)

        dataset = self.dataset_frame([frame], time_index)

        if 'orderbook' not in self.dataset:
            self.dataset = dataset
        else:
            self.dataset = xr.concat((self.dataset, dataset), dim='time')

        self.publish_last_frame()
        self.to_netcdf()

        return self.dataset

    def publish_last_frame(self):
        last_index = self.dataset.time[-2:]
        dataset = self.dataset.sel(time=slice(*last_index.data))
        last_frame_values: ndarray = dataset.orderbook.values[-1]

        self.ws_emit(
            self.frame_channel,
            json.dumps(last_frame_values.tolist())
        )


@click.command()
@click.argument('symbol', type=click.Choice(BitmexChannels.__members__))
@click.option('--save-interval', '-i', default='1h', help='save interval as string "1h"')
@click.option('--no-save', '-n', is_flag=True, help='disable saving to disk')
@click.option('--enable-memory-tracing', '-t', is_flag=True,
              help='Enable memory tracing.')
def main(symbol: str, save_interval: str, no_save: bool, **kwargs):
    kwargs['save'] = not no_save

    recorder = BitmexOrderBookEmitter(
        symbol=BitmexChannels[symbol],
        save_interval=save_interval,
        **kwargs
    )

    signal.signal(signal.SIGINT, recorder.exit)
    signal.signal(signal.SIGTERM, recorder.exit)

    recorder.start()


if __name__ == '__main__':
    main()
