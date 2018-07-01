from exchange_data.bitmex_orderbook import BitmexOrderBook
from exchange_data.hdf5_orderbook import Hdf5OrderBook

import alog
import datetime
import numpy as np


class BitmexOrderBookGymData(BitmexOrderBook, Hdf5OrderBook):
    def __init__(self,
                 symbol: str,
                 overwrite: bool,
                 cache_dir: str = None,
                 total_time: str = '15m',
                 read_from_json: bool = False,
                 interval='1m'):
        database = 'bitmex'
        Hdf5OrderBook.__init__(self,
                               database=database,
                               symbol=symbol,
                               overwrite=overwrite,
                               cache_dir=cache_dir,
                               read_from_json=read_from_json,
                               total_time=total_time)

        BitmexOrderBook.__init__(self, symbol=symbol)

        self.interval = interval
        self.current_time = None
        self._last_index = 0

    @property
    def dataset_name(self):
        return f'{self.database}_{self.symbol}_{self.interval}'

    @property
    def half_spread(self):
        return self.spread / 2

    def fetch_and_save(self):
        self.fetch_measurements()

        for line in self.result_set['data']:
            self.replay(line)

        self.file.close()

    def replay(self, line):
        msg = self.message(line)
        self.set_time(msg.timestamp)

    def set_time(self, timestamp):
        _time = datetime.datetime.fromtimestamp(timestamp / 1000)
        _time = _time.replace(second=0, microsecond=0)
        prev_time = self.current_time
        self.current_time = _time

        if prev_time != self.current_time:
            self.next_tick()

    def next_tick(self):
        try:
            self.save_frame()
        except IndexError:
            pass

    def save_frame(self):
        orderbook_frame = self.gen_orderbook_frame()
        self.storage.append(orderbook_frame)

    def gen_orderbook_frame(self):
        bid_side = self.gen_bid_side()
        bid_side.resize((2000, 2))
        bid_side -= self.half_spread

        ask_side = self.gen_ask_side()
        ask_side.resize((2000, 2))
        ask_side += self.half_spread

        frame = np.array([ask_side, bid_side])
        return frame

    def gen_bid_side(self):
        bid_levels = list(self.bids.price_tree.items(reverse=True))
        price = np.array(
            [level[0] for level in bid_levels]
        )
        volume = np.array(
            [level[1].volume for level in bid_levels]
        )
        max_bid_diff = price - price[0]
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
        max_ask_diff = price - price[0]
        volume_pct = np.true_divide(volume, np.sum(volume))
        return np.vstack((max_ask_diff, volume_pct))

    def dataset(self, file):
        if self._dataset:
            return self._dataset
        elif self.dataset_name in file:
            return file[self.dataset_name]
        else:
            return file.create_dataset(self.dataset_name, (0, 0, 0, 0),
                                       dtype=float, chunks=True,
                                       maxshape=(float('1e5'), float('1e5'),
                                                 float('1e5'), float('1e5'),))


if __name__ == '__main__':
    pass
