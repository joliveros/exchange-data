from abc import ABC
from cached_property import cached_property
from collections import Generator
from datetime import datetime, timedelta
from dateutil import parser
from dateutil import parser
from dateutil.tz import tz
from exchange_data import Database, settings
from exchange_data.emitters import SignalInterceptor
from exchange_data.utils import random_date, DateTimeUtils, NoValue
from numpy.core.multiarray import ndarray
from pandas import to_datetime, DataFrame
from pytimeparse.timeparse import timeparse
from time import sleep
from typing import Tuple
from xarray import Dataset, DataArray

import alog
import click
import json
import numpy as np
import re
import traceback

alog.set_level(settings.LOG_LEVEL)

class OutOfFramesException(Exception):
    pass


class BitmexOrderBookChannels(NoValue):
    XBTUSD = 'XBTUSD_OrderBookFrame'


class BitmexStreamer(Database, Generator, DateTimeUtils, SignalInterceptor, ABC):
    def __init__(
        self,
        max_spread: float = 100.0,
        orderbook_depth: int = 10,
        random_start_date: bool = False,
        end_date: datetime = None,
        start_date: datetime = None,
        window_size: str = '15s',
        sample_interval: str = '1s',
        channel_name: str = None,
        **kwargs
    ):
        super().__init__(database_name='bitmex', **kwargs)
        SignalInterceptor.__init__(self)

        self.random_start_date = random_start_date
        self.out_of_frames_counter = 0
        self.sample_interval = sample_interval
        self.sample_interval_s = timeparse(sample_interval)
        self._time = []
        self._index = []
        self.end_date = None
        self.max_spread = max_spread
        self.start_date = None
        self.realtime = False
        self._min_date = parser.parse('2018-06-02 22:49:31.148000+00:00')

        self.orderbook_depth = orderbook_depth
        self.window_size = timeparse(window_size)

        if channel_name:
            self.channel_name = channel_name
        else:
            self.channel_name = BitmexOrderBookChannels.XBTUSD.value

        if self.random_start_date:
            if start_date is not None:
                raise Exception('start_date should be None.')

            self.start_date = random_date(self.min_date, self.now())

        if start_date is not None:
            self.start_date = start_date
        elif self.start_date is None:
            self.start_date = self.now() - timedelta(seconds=self.window_size)
            self.realtime = True
        if end_date is not None:
            self.end_date = end_date
        else:
            self.end_date = self.start_date + \
                            timedelta(seconds=self.window_size)

        if self.start_date < self.min_date:
            raise Exception('Start date not available in DB.')

        self.window_size = (self.end_date - self.start_date).total_seconds()

        # alog.info(self.realtime)
        # alog.info(self.start_date)
        # alog.info(self.end_date)
        #
        # raise Exception()

    @cached_property
    def min_date(self):
        if self._min_date:
            return self._min_date
        start_date = datetime.fromtimestamp(0, tz=tz.tzutc())

        if self.end_date is None:
            self.end_date = self.now()

        result = self.oldest_frame_query(start_date, self.end_date)

        for item in result.get_points(self.channel_name):
            timestamp = datetime.utcfromtimestamp(item['time']/1000)\
                .replace(tzinfo=tz.tzutc())
            return timestamp

        raise Exception('Database has no data.')

    def orderbook_frames(self):
        return self._orderbook_frames(self.start_date, self.end_date)

    def oldest_frame_query(self, start_date, end_date):
        start_date = self.format_date_query(start_date)
        end_date = self.format_date_query(end_date)

        query = f'SELECT * FROM {self.channel_name} ' \
            f'WHERE time > {start_date} AND ' \
            f'time < {end_date} LIMIT 1 tz(\'UTC\');'

        return self.query(query)

    def orderbook_frame_query(self, start_date=None, end_date=None):
        start_date = start_date if start_date else self.start_date
        end_date = end_date if end_date else self.end_date

        start_date = self.format_date_query(start_date)
        end_date = self.format_date_query(end_date)
        query = f'SELECT LAST(data) as data FROM {self.channel_name} ' \
            f'WHERE time > {start_date} AND time < {end_date} ' \
            f'GROUP BY time({self.sample_interval}) tz(\'UTC\');'
        # query = f'SELECT * FROM {self.channel_name} ' \
        #     f'WHERE time >= {start_date} AND time < {end_date} ' \
        #     f'tz(\'UTC\');'

        return self.query(query)

    def parse_db_timestamp(self, timestamp):
        return datetime.utcfromtimestamp(timestamp / 1000)\
                .replace(tzinfo=tz.tzlocal())

    def _orderbook_frames(self, start_date, end_date):
        orderbook = self.orderbook_frame_query(start_date, end_date)
        frame_list = []
        max_shape = (0, 0, 0)
        time_index = []

        for item in orderbook.get_points(self.channel_name):
            time_index.append(self.parse_db_timestamp(item['time']))
            data = np.asarray(json.loads(item['data']), dtype=np.float32)

            if data.shape[-1] > max_shape[-1]:
                max_shape = data.shape

            frame_list.append(data)

        time_index.reverse()
        frame_list.reverse()

        if len(frame_list) == 0:
            self.out_of_frames_counter += 1
            raise OutOfFramesException

        resized_frames = []
        for frame in frame_list:
            if frame.shape[-1] != max_shape[-1]:
                new_frame = np.zeros(max_shape)
                shape = frame.shape
                new_frame[:shape[0], :shape[1], :shape[2]] = frame
                resized_frames.append(new_frame)
            else:
                resized_frames.append(frame)

        window = np.stack(resized_frames, axis=0)

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
            f'WHERE time > {start_date} AND ' \
            f'time < {end_date} tz(\'UTC\');'

        index = self.query(query)

        index = [item for item in index.get_points('.BXBT_1m')]
        index.reverse()

        return index

    def compose_window(self) -> Tuple[ndarray, ndarray, ndarray]:
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

        return time_index, index, orderbook.to_array().values[0][:, :, :, :self.orderbook_depth]

    def next_window(self):
        result = np.array([]), np.array([]), np.array([])

        now = self.now()
        if self.realtime:
            self.start_date = now - timedelta(seconds=self.window_size + self.sample_interval_s)
            self.end_date = now

        result = self.compose_window()

        if not self.realtime:
            self._set_next_window()

        return result

    def _set_next_window(self):
        self.start_date += timedelta(
            seconds=self.window_size + self.sample_interval_s)
        self.end_date += timedelta(seconds=self.window_size)

    def send(self, *args):
        while len(self._index) == 0:
            time, index, orderbook = self.next_window()
            self._time += time.tolist()
            self._index += index.tolist()
            self._orderbook += orderbook.tolist()

        return self._time.pop(), self._index.pop(), self._orderbook.pop()

    def throw(self, type=None, value=None, traceback=None):
        raise StopIteration


@click.command()
@click.option('--random-start-date',
              '-r',
              is_flag=True,
              help='Enable random start date.')
@click.option('--window-size',
              '-w',
              type=str,
              default='1m',
              help='Window size i.e. "1m"')
@click.option('--sample-interval',
              '-s',
              type=str,
              default='1s',
              help='interval at which to sample data from db.')
def main(**kwargs):
    streamer = BitmexStreamer(**kwargs)

    while True:
        timestamp, index, orderbook = next(streamer)
        orderbook_ar = np.array(orderbook)

        alog.info('\n'+ str(datetime.fromtimestamp(timestamp/10**9)) +'\n' +
                  str(index) + '\n' + str(orderbook_ar[:, :, :1]))
        sleep(0.1)


if __name__ == '__main__':
    main()
