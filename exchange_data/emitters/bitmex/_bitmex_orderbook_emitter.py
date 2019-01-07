from exchange_data import settings
from exchange_data.bitmex_orderbook import BitmexOrderBook
from exchange_data.emitters import Messenger, TimeChannels
from exchange_data.emitters.bitmex import BitmexEmitterBase, BitmexChannels
from numpy.core.multiarray import ndarray
from pandas import date_range
from pytimeparse import parse as dateparse
from xarray import Dataset, DataArray

import alog
import click
import numpy as np
import signal
import sys
import xarray as xr


class BitmexOrderBookEmitter(
    BitmexEmitterBase,
    Messenger,
    BitmexOrderBook
):
    def __init__(self, symbol: BitmexChannels, max_dataset_length='1d'):
        BitmexOrderBook.__init__(self, symbol)
        Messenger.__init__(self)
        BitmexEmitterBase.__init__(self, symbol)

        self.max_dataset_length = dateparse(max_dataset_length)
        self.freq = settings.TICK_INTERVAL
        self.dataset: Dataset = None

        self.on(TimeChannels.Tick.value, self.update_dataset)
        self.on(self.symbol.value, self.message)

    def start(self):
        self.sub([self.symbol, TimeChannels.Tick])

    def stop(self, *args):
        self._pubsub.close()
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

        if dataset:
            prev_frame_shape = \
                dataset.sel(time=dataset.time[-1]).orderbook.shape

        bid_side = self.gen_bid_side()
        ask_side = self.gen_ask_side()

        frame = np.append(ask_side, bid_side, axis=1)

        if frame.shape[-1] > prev_frame_shape[-1]:
            if dataset:
                self.resize_dataset(dataset, frame)

        elif frame.shape[-1] < prev_frame_shape[-1]:
                book_shape = self.dataset.orderbook.values.shape
                shape = (book_shape[1], book_shape[2])
                resized_values = np.zeros(shape)
                frame_shape = frame.shape

                resized_values[:frame_shape[0], :frame_shape[1]] = frame
                return resized_values

        return frame

    def resize_dataset(self, dataset, frame):
        book_shape = self.dataset.orderbook.values.shape
        resized_values = np.zeros((book_shape[0],) + frame.shape)

        resized_values[
            :book_shape[0],
            :book_shape[1],
            :book_shape[2]
        ] = dataset.orderbook.values

        self.dataset = self.dataset_frame(
            resized_values,
            dataset.time
        )

    @staticmethod
    def dataset_frame(values, time):
        return Dataset(
            {
                'orderbook': (['time', 'price', 'volume'], values)
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

        if self.dataset is None:
            self.dataset = dataset
        else:
            self.dataset = xr.concat((self.dataset, dataset), dim='time')

        self.trim_dataset()

        return self.dataset

    def trim_dataset(self):
        if len(self.dataset.time) > self.max_dataset_length:
            new_index = self.dataset.time[self.max_dataset_length * -1:].data

            self.dataset = self.dataset.sel(time=slice(*new_index))


@click.command()
@click.argument('symbol', type=click.Choice(BitmexChannels.__members__))
def main(symbol: str):

    recorder = BitmexOrderBookEmitter(symbol=BitmexChannels[symbol])

    signal.signal(signal.SIGINT, recorder.stop)
    signal.signal(signal.SIGTERM, recorder.stop)

    recorder.start()


if __name__ == '__main__':
    main()
