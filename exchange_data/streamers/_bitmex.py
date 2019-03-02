from datetime import datetime, timedelta
from exchange_data import Database
from pandas import to_datetime, DataFrame
from pytimeparse.timeparse import timeparse
from xarray import Dataset, DataArray

import alog
import json
import numpy as np
import pytz
import xarray as xr


class BitmexStreamer(Database):
    def __init__(
        self,
        start_date: float = None,
        window_size: str = '1m',
        depth: int = 10,
        **kwargs
    ):
        super().__init__(database_name='bitmex', **kwargs)

        self.depth = depth
        self.window_size = timeparse(window_size)

        if start_date is None:
            start_date = datetime.utcnow()

        self._start_date = start_date.replace(tzinfo=pytz.UTC)
        self._end_date = self._start_date + timedelta(seconds=self.window_size)

    @property
    def start_date(self):
        return f'\'{self._start_date.isoformat()}\''

    @property
    def end_date(self):
        return f'\'{self._end_date.isoformat()}\''

    def orderbook_frames(self):
        query = f'SELECT * FROM XBTUSD_OrderBookFrame ' \
            f'WHERE time > {self.start_date} AND time < {self.end_date};'

        orderbook = self.query(query)

        window_list = []
        max_shape = (0, 0, 0)
        time_index = []

        for item in orderbook.get_points('XBTUSD_OrderBookFrame'):
            time_index.append(item['time'] * 1000 * 1000)
            data = np.asarray(json.loads(item['data']), dtype=np.float32)
            window_list.append(data)

            if data.shape[-1] > max_shape[-1]:
                max_shape = data.shape

        window_list = [self.resize_frame(data, max_shape) for data in window_list]

        window = np.stack(window_list, axis=0)

        orderbook: Dataset = Dataset({
            'orderbook': (['time', 'side', 'level', 'price'], window)
        }, coords={
            'time': to_datetime(time_index, utc=True)
        })

        return orderbook

    def index(self) -> np.ndarray:
        query = f'SELECT * FROM ".BXBT_1m" ' \
            f'WHERE time > {self.start_date} AND time < {self.end_date};'

        index = self.query(query)

        return [item for item in index.get_points('.BXBT_1m')]

    def compose_window(self):
        orderbook = self.orderbook_frames()

        time_index: DataArray = orderbook.time.values.copy()
        _index = [item for item in self.index() if item['weight'] is not None]
        timestamps = set([item['timestamp'] for item in _index])

        price_timestamp = [
            (
                timestamp * 10 ** 6, [index['lastPrice'] for index in _index
                                      if index['timestamp'] == timestamp]
            ) for timestamp in timestamps
        ]

        avg_prices = [(pt[0], sum(pt[1]) / float(len(pt[1])))
                      for pt in price_timestamp]

        index: DataFrame = DataFrame(
            {'avg_price': np.NaN},
            index=to_datetime(time_index, utc=True)
        )

        for price in avg_prices:
            timestamp, avg_price = price
            ts_idx = index.index.get_loc(timestamp, method='nearest')
            index.avg_price[ts_idx] = avg_price

        index = index.fillna(method='ffill')
        index['index_diff'] = index.diff()

        index_ds = Dataset.from_dataframe(index) \
            .rename({'index': 'time'}) \
            .fillna(0)

        orderbook = orderbook.diff('price')

        price_info = xr.merge([index_ds, orderbook])

        orderbook.close()

        return price_info.sel(
            price=slice(None, self.depth)
        ).to_array().values

    def __iter__(self):
        pass

    def resize_frame(self, data, max_shape):
        frame = np.zeros(max_shape)
        frame[
        :data.shape[0],
        :data.shape[1],
        :data.shape[2]
        ] = data
        return frame
