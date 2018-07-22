from exchange_data.bitmex_orderbook import BitmexOrderBook
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

    def nearest_date(self):
        last_datetime = datetime_from_timestamp(self.last_timestamp)
        # nearest_date = min(self.dataset.time)

        # round to nearest freq


    def init_dataset(self):
        time_index = date_range(
            start=self.date_range['start'],
            end=self.date_range['end'],
            freq=self.freq)
        self.dataset['time'] = time_index

        alog.debug(time_index)

        volume_pct_shape = (time_index.shape[0], self.max_levels)
        volume_pct_data = np.zeros(volume_pct_shape)
        self.dataset['bid_volume_pct'] = \
            (['time', 'volume_pct'], volume_pct_data)

        # alog.debug(self.dataset)

    def fetch_and_save(self):
        self.fetch_measurements()

        for line in self.result_set['data']:
            self.replay(line)

        self.save()

    def replay(self, line):
        msg = self.message_strict(line)

        if self.date_range is None:
            self.read_date_range(msg)
            self.init_dataset()

        self.save_frame()

    def save_frame(self):
        pass
        # alog.debug(self.bid_volume_by_percent)
        # alog.debug(self.dataset.volume_pct['time'][self.last_timestamp])

        # bid_side.resize((2000, 2))
        # bid_side -= self.half_spread
        #
        # ask_side = self.gen_ask_side()
        # ask_side.resize((2000, 2))
        # ask_side += self.half_spread

        # frame = np.array([ask_side, bid_side])

        # alog.debug(frame)

        # return frame

    def gen_bid_side(self):
        price = np.array(
            [level[0] for level in self.sorted_bid_levels]
        )

        if len(price) == 0:
            max_bid_diff = price
        else:
            max_bid_diff = price - price[0]

        return max_bid_diff

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
        self.date_range = {
            'start': msg.timestamp_datetime,
            'end': date_plus_timestring(msg.timestamp, self.total_time)
        }


if __name__ == '__main__':
    pass
