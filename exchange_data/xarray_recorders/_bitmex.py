
from exchange_data import settings
from exchange_data.bitmex_orderbook import BitmexOrderBook
from exchange_data.cached_dataset import CachedDataset
from exchange_data.emitters import Messenger, TimeChannels, BitmexChannels, \
    TimeEmitter
from exchange_data.utils import NoValue, date_plus_timestring, \
    datetime_from_timestamp
from numpy.core.multiarray import ndarray
from pandas import date_range
from xarray import Dataset

import decimal
import signal
import alog
import click
import numpy as np
import xarray as xr


class BitmexEvent(NoValue):
    orderBookL2 = 'orderBookL2'


class BitmexXArrayRecorder(Messenger, BitmexOrderBook, CachedDataset):
    def __init__(
        self,
        symbol: BitmexChannels,
        overwrite: bool = False,
        cache_dir: str = None,
        max_levels=1000,
        total_time: str = '1m'
    ):

        self.previous_day: decimal = TimeEmitter().previous_day

        BitmexOrderBook.__init__(self, symbol=symbol)
        Messenger.__init__(self)
        CachedDataset.__init__(self, cache_dir=cache_dir, overwrite=overwrite)

        self.tick_counter = 0
        self._last_index = 0
        self.current_time = None
        self.database = 'bitmex'
        self.date_range = None
        self.freq = settings.TICK_INTERVAL
        self.max_levels = max_levels
        self.symbol: BitmexChannels = symbol
        self.total_time = total_time
        self.prefix = self.symbol.value

        self.on(TimeChannels.Tick.value, self.frame)
        self.on(TimeChannels.NextDay.value, self.next_day)
        self.on(self.symbol.value, self.message)

    @property
    def dataset_name(self):
        return f'{self.database}_{self.symbol}_{self.interval}'

    @property
    def filename(self):
        return f'{self.cache_dir}/{self.prefix}_{int(self.previous_day)}.{self.extension}'

    @property
    def half_spread(self):
        return self.spread / 2

    @property
    def bid_volume_by_percent(self):
        volume_pct = np.true_divide(self.bid_volume, np.sum(self.bid_volume))
        return volume_pct

    @property
    def bid_volume(self):
        volume = np.array(
            [level[1].volume for level in self.sorted_bid_levels]
        )
        return volume

    @property
    def sorted_bid_levels(self):
        return list(self.bids.price_tree.items(reverse=True))

    def read_date_range(self):
        if self.date_range:
            raise Exception('date_range has already been set.')

        self.date_range = {
            'start': datetime_from_timestamp(self.last_timestamp),
            'end': date_plus_timestring(self.last_timestamp, self.total_time)
        }

    def init_dataset(self):
        time_index = date_range(
            start=self.last_date,
            end=self.last_date,
            freq=self.freq
        ).floor(self.freq)

        dataset = Dataset()

        dataset['time'] = time_index

        volume_pct_shape = (time_index.shape[0], 2, 2, self.max_levels)
        volume_pct_data = np.zeros(volume_pct_shape)

        dataset['orderbook'] = \
            (['time', 'side', 'price', 'volume'], volume_pct_data)

        return dataset

    def gen_bid_side(self):
        bid_levels = list(self.asks.price_tree.items())
        price = np.array(
            [level[0] for level in bid_levels]
        )

        if len(price) == 0:
            max_bid_diff = price
        else:
            max_bid_diff = price - price[0]

        volume = np.array(
            [level[1].volume for level in bid_levels]
        )
        volume_pct = np.true_divide(volume, np.sum(volume))

        return np.vstack((max_bid_diff, volume_pct))

    def gen_ask_side(self):
        ask_levels = list(self.asks.price_tree.items())

        price = np.array(
            [level[0] for level in ask_levels]
        )
        volume = np.array(
            [level[1].volume for level in ask_levels]
        )

        if len(price) == 0:
            max_ask_diff = price
        else:
            max_ask_diff = price - price[0]

        volume_pct = np.true_divide(volume, np.sum(volume))

        return np.vstack((max_ask_diff, volume_pct))

    def generate_frame(self) -> ndarray:
        bid_side = self.gen_bid_side()
        bid_side.resize((2, self.max_levels))
        bid_side -= self.half_spread

        ask_side = self.gen_ask_side()
        ask_side.resize((2, self.max_levels))
        ask_side += self.half_spread

        frame = np.array([ask_side, bid_side])

        # alog.info(alog.pformat(frame))

        return frame

    def frame(self, timestamp) -> ndarray:
        self.last_timestamp = timestamp

        next_dataset = self.init_dataset()

        frame = self.generate_frame()

        nearest_date = next_dataset.sel(time=self.last_date,
                                        method='nearest').time.values

        next_dataset['orderbook'].loc[dict(time=nearest_date)] = frame

        if self.dataset.dims.keys() == next_dataset.dims.keys():
            self.dataset = xr.concat([self.dataset, next_dataset], 'time')
        else:
            self.dataset = next_dataset

        # alog.info(self.dataset.orderbook.shape)

        self.to_netcdf()

        return frame

    def to_netcdf(self):
        self.tick_counter += 1
        if self.tick_counter % 60 == 0:
            alog.info('### saving ###')
            super().to_netcdf()

    def next_day(self, timestamp):
        self.previous_day = timestamp
        self.to_netcdf()
        self.dataset = Dataset()

    def start(self):
        self.sub([TimeChannels.Tick, self.symbol])

    def stop(self, *args):
        alog.info('### saving ###')
        super().to_netcdf()
        exit(0)


@click.command()
@click.argument('symbol', type=click.Choice(BitmexChannels.__members__))
def main(symbol: str):

    recorder = BitmexXArrayRecorder(
        symbol=BitmexChannels[symbol]
    )

    signal.signal(signal.SIGINT, recorder.stop)
    signal.signal(signal.SIGTERM, recorder.stop)

    recorder.start()


if __name__ == '__main__':
    main()
