from datetime import timedelta, datetime
from exchange_data import Measurement, Database
from exchange_data.utils import DateTimeUtils
from pandas import DataFrame
from pytimeparse.timeparse import timeparse
from copy import copy
import alog
import json
import pandas as pd
import re


class MeasurementMeta(Database, DateTimeUtils):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def name(self):
        raise NotImplemented()


class MeasurementFrame(MeasurementMeta):
    def __init__(
        self,
        interval='1h',
        group_by='1m',
        start_date=None,
        end_date=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self._start_date = None
        self._end_date = None
        self.group_by = group_by
        self.interval_str = interval

        self.interval = timedelta(seconds=timeparse(interval))

        self.start_date = DateTimeUtils.now() - self.interval
        self.end_date = DateTimeUtils.now()

        self.original_interval = (copy(self.start_date), copy(self.end_date))

    def reset_interval(self):
        self.start_date = self.original_interval[0]
        self.end_date = self.original_interval[1]

    @property
    def name(self):
        return re.sub(r'(?<!^)(?=[A-Z])', '_', type(self).__name__).lower()

    @property
    def formatted_start_date(self):
        return self.format_date_query(self.start_date)

    @property
    def formatted_end_date(self):
        return self.format_date_query(self.end_date)

    @property
    def start_date(self):
        return self._start_date
    @start_date.setter
    def start_date(self, value):
        if value is None:
            raise Exception()
        self._start_date = value
    @property
    def end_date(self):
        return self._end_date
    @end_date.setter
    def end_date(self, value):
        if value is None:
            raise Exception()
        self._end_date = value

    def frame(self):
        query = f'SELECT first(*) AS data FROM {self.name} WHERE time >=' \
                f' {self.formatted_start_date} AND ' \
                f'time <= {self.formatted_end_date} GROUP BY time(' \
                f'{self.group_by})'

        alog.info(query)

        frames = []

        for data in self.query(query).get_points(self.name):
            timestamp = self.parse_db_timestamp(data['time'])
            data = data.get('data_data', None) or {}

            if type(data) is str:
                data = pd.read_json(data)

            if 'pair' in data:
                data = json.loads(data)
                pair = data['pair']
                pair = dict(map(reversed, pair.items()))

                data = dict(
                    time=timestamp,
                    **pair
                )

                frames.append(data)

        df = DataFrame.from_dict(frames)

        if df.empty:
            return df

        df['time'] = pd.to_datetime(df['time'])

        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)

        return df

    def frame_all_keys(self):
        query = f'SELECT first("short_period") AS short_period, ' \
                f'first("long_period") as long_period, first("group_by_min") ' \
                f'as group_by_min, first("symbol") as symbol, first("value") ' \
                f'as value FROM' \
                f' {self.name} WHERE time >=' \
                f' {self.formatted_start_date} AND ' \
                f'time <= {self.formatted_end_date} GROUP BY time(' \
                f'{self.group_by})'

        frames = []

        for data in self.query(query).get_points(self.name):
            timestamp = self.parse_db_timestamp(data['time'])
            data['time'] = timestamp

            frames.append(data)

        df = DataFrame.from_dict(frames)

        df['time'] = pd.to_datetime(df['time'])

        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)
        df.dropna(subset=['value'], inplace=True)

        return df

    def append(self, data: dict, timestamp: datetime = None):
        if timestamp is None:
            timestamp = self.now()

        m = Measurement(
            measurement=self.name,
            fields=data,
            time=timestamp
        )

        self.write_points([m.__dict__])
