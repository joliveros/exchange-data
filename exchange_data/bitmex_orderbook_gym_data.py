from numpy.core.multiarray import ndarray

from exchange_data.bitmex_orderbook import BitmexOrderBook, NotOrderbookMessage
from exchange_data.cached_dataset import CachedDataset
from exchange_data.influxdb_data import InfluxDBData
from exchange_data.utils import date_plus_timestring, datetime_from_timestamp
from pandas import date_range

import alog
import numpy as np


class BitmexOrderBookGymData(BitmexOrderBook, CachedDataset, InfluxDBData):
    def __init__(self,
                 symbol: str,
                 overwrite: bool,
                 cache_dir: str = None,
                 total_time: str = '15m',
                 read_from_json: bool = False,
                 max_levels=1000
                 ):
        self.freq = '1s'

        self.max_levels = max_levels

        self.date_range = None

        self.total_time = total_time

        self.database = 'bitmex'

        self.current_time = None

        self._last_index = 0

        CachedDataset.__init__(
            self,
            overwrite=overwrite,
            cache_dir=cache_dir
        )

        BitmexOrderBook.__init__(self, symbol=symbol)

        InfluxDBData.__init__(
            self,
            database=self.database,
            read_from_json=read_from_json,
            symbol=symbol,
            total_time=total_time
        )

    @property
    def dataset_name(self):
        return f'{self.database}_{self.symbol}_{self.interval}'

    @property
    def filename(self):
        return f'{self.cache_dir}/{self.storage_name}_{self.total_time}.' \
               f'{self.extension} '

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

    def init_dataset(self):
        time_index = date_range(
            start=self.date_range['start'],
            end=self.date_range['end'],
            freq=self.freq
        )

        self.dataset['time'] = time_index.round('s')

        volume_pct_shape = (time_index.shape[0], 2, 2, self.max_levels)
        volume_pct_data = np.zeros(volume_pct_shape)

        self.dataset['orderbook'] = \
            (['time', 'side', 'price', 'volume'], volume_pct_data)

    def fetch_and_save(self):
        self.fetch_measurements()

        for line in self.result_set.get_points():
            self.replay(line)

        self.save()

    def replay(self, line):
        try:
            msg = self.message(line)

            if self.date_range is None:
                self.read_date_range(msg)
                self.init_dataset()

            self.add_frame()
        except NotOrderbookMessage:
            pass

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

    def read_date_range(self, msg):
        if self.date_range:
            raise Exception('date_range has already been set.')

        self.date_range = {
            'start': msg.timestamp_datetime,
            'end': date_plus_timestring(msg.timestamp, self.total_time)
        }

    def generate_frame(self) -> ndarray:
        bid_side = self.gen_bid_side()
        bid_side.resize((2, self.max_levels))
        bid_side -= self.half_spread
        ask_side = self.gen_ask_side()
        ask_side.resize((2, self.max_levels))
        ask_side += self.half_spread
        frame = np.array([ask_side, bid_side])
        return frame

    def add_frame(self) -> ndarray:
        frame = self.generate_frame()

        if self.last_date not in self.dataset.time.values:
            nearest_date = self.dataset.sel(time=self.last_date, method='nearest').time.values

            self.dataset['orderbook'].loc[dict(time=nearest_date)] = frame

        return frame

