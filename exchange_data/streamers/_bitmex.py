from collections import Generator
from time import sleep

import click
from cached_property import cached_property
from datetime import datetime, timedelta
from dateutil.tz import tz
from exchange_data import Database, settings
from exchange_data.utils import random_date
from numpy.core.multiarray import ndarray
from pandas import to_datetime, DataFrame
from pytimeparse.timeparse import timeparse
from typing import Tuple
from xarray import Dataset, DataArray

import alog
import json
import numpy as np

alog.set_level(settings.LOG_LEVEL)


class BitmexStreamer(Database, Generator):
    def __init__(
        self,
        depth: int = 10,
        random_start_date: bool = False,
        end_date: datetime = None,
        start_date: datetime = None,
        window_size: str = '1m',
        **kwargs
    ):
        super().__init__(database_name='bitmex', **kwargs)
        self.start_date = None
        self.end_date = None
        self._index = []
        self._orderbook = []

        self.depth = depth
        self.window_size = timeparse(window_size)
        self.channel_name = 'XBTUSD_OrderBookFrame'

        if random_start_date:
            if start_date is not None:
                raise Exception('start_date should be None.')

            self.start_date = random_date(self.min_date, self.now())

        if self.start_date is None:
            self.start_date = start_date
        if self.start_date is None:
            self.start_date = self.now()

        if end_date is not None:
            self.end_date = end_date
        else:
            self.end_date = self.start_date + \
                            timedelta(seconds=self.window_size)

        if self.start_date < self.min_date:
            raise Exception('Start date not available in DB.')

        self.window_size = (self.end_date - self.start_date).total_seconds()

    def now(self):
        return datetime.now(tz=tz.tzlocal())

    @cached_property
    def min_date(self):
        start_date = datetime.fromtimestamp(0, tz=tz.tzlocal())

        if self.end_date is None:
            self.end_date = self.now()

        result = self.oldest_frame_query(start_date, self.end_date)

        for item in result.get_points(self.channel_name):
            return datetime.utcfromtimestamp(item['time']/1000)\
                .replace(tzinfo=tz.tzlocal())

        raise Exception('Database has no data.')

    def format_date_query(self, value):
        return f'\'{value.replace(tzinfo=None)}\''

    def orderbook_frames(self):
        return self._orderbook_frames(self.start_date, self.end_date)

    def oldest_frame_query(self, start_date, end_date):
        start_date = self.format_date_query(start_date)
        end_date = self.format_date_query(end_date)

        query = f'SELECT * FROM {self.channel_name} ' \
            f'WHERE time > {start_date} AND time < {end_date} LIMIT 1;'

        alog.debug(query)

        return self.query(query)

    def orderbook_frame_query(self, start_date=None, end_date=None):
        start_date = start_date if start_date else self.start_date
        end_date = end_date if end_date else self.end_date

        start_date = self.format_date_query(start_date)
        end_date = self.format_date_query(end_date)

        query = f'SELECT * FROM {self.channel_name} ' \
            f'WHERE time > {start_date} AND time < {end_date};'

        return self.query(query)

    def parse_db_timestamp(self, timestamp):
        return datetime.utcfromtimestamp(timestamp / 1000)\
                .replace(tzinfo=tz.tzlocal())

    def _orderbook_frames(self, start_date, end_date):
        orderbook = self.orderbook_frame_query(start_date, end_date)

        window_list = []
        max_shape = (0, 0, 0)
        time_index = []

        for item in orderbook.get_points(self.channel_name):
            time_index.append(self.parse_db_timestamp(item['time']))
            data = np.asarray(json.loads(item['data']), dtype=np.float32)

            buy_side = data[1, :, :]
            zeros = (buy_side == 0.0).argmax(axis=1)
            buy_side = np.flip(buy_side[:, :zeros[0]], 1)[:, : self.depth]

            data = data[:, :, :self.depth]
            data[1, :buy_side.shape[0], :buy_side.shape[1]] = buy_side
            window_list.append(data)

        window = np.stack(window_list, axis=0)

        orderbook: Dataset = Dataset({
            'orderbook': (['time', 'side', 'level', 'price'], window)
        }, coords={
            'time': to_datetime(time_index, utc=True)
        })
        return orderbook

    def index(self) -> list:
        start_date = self.format_date_query(self.start_date)
        end_date = self.format_date_query(self.end_date)

        range_diff = (self.end_date - self.start_date).total_seconds()

        if range_diff < timeparse('1m'):
            start_date = self.format_date_query(
                self.end_date - timedelta(seconds=timeparse('1m'))
            )

        query = f'SELECT * FROM ".BXBT_1m" ' \
            f'WHERE time > {start_date} AND time < {end_date};'

        index = self.query(query)

        return [item for item in index.get_points('.BXBT_1m')]

    def compose_window(self) -> Tuple[ndarray, ndarray]:
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

        index: ndarray = Dataset.from_dataframe(index) \
            .rename({'index': 'time'}) \
            .fillna(0).to_array().values[0]

        return index, orderbook.to_array().values[0]

    def next_window(self):
        self.start_date += timedelta(seconds=self.window_size)
        self.end_date += timedelta(seconds=self.window_size)
        return self.compose_window()

    def send(self, *args):
        if len(self._index) == 0:
            index, orderbook = self.next_window()
            self._index += index.tolist()
            self._orderbook += orderbook.tolist()

        return self._index.pop(), self._orderbook.pop()

    def throw(self, type=None, value=None, traceback=None):
        raise StopIteration


@click.command()
@click.option('--random-start-date',
              '-r',
              is_flag=True,
              help='Enable random start date.')
def main(**kwargs):
    streamer = BitmexStreamer(**kwargs)

    while True:
        index, orderbook = next(streamer)
        orderbook_ar = np.array(orderbook)
        alog.info(orderbook_ar)
        sleep(1)


if __name__ == '__main__':
    main()
