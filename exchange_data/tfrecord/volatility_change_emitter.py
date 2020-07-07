#!/usr/bin/env python
from datetime import timedelta, datetime

import pandas
from pandas import DataFrame
from pytimeparse.timeparse import timeparse

from exchange_data import Database, Measurement
import alog
import click
import numpy as np
import re
import pandas as pd

from exchange_data.emitters import Messenger
from exchange_data.utils import DateTimeUtils

pd.options.plotting.backend = 'plotly'


class MeasurementMeta(Database, DateTimeUtils):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def name(self):
        raise NotImplemented()


class MeasurementFrame(MeasurementMeta):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def name(self):
        return re.sub(r'(?<!^)(?=[A-Z])', '_', type(self).__name__).lower()

    def append(self, data: dict, timestamp: datetime = None):
        if timestamp is None:
            timestamp = self.now()

        m = Measurement(
            measurement=self.name,
            fields=data,
            time=timestamp
        )

        alog.info(alog.pformat(m))


class VolatilityChange(MeasurementFrame):
    def __init__(self, interval, group_by, filter, **kwargs):
        super().__init__(
            batch_size=1,
            **kwargs)

        self.group_by = group_by
        self.filter = filter
        self.interval_delta = timedelta(seconds=timeparse(interval))

    def tick(self):
        self.append(self.volatility_change())

    def volatility_change(self):
        self.frames = self.get_frames()
        # alog.info(alog.pformat(self.frames))
        volatility_change = []
        for df in self.frames:
            df['log_ret'] = np.log(df['price'] / df['price'].shift(1))
            df.dropna(how='any', inplace=True)

            window = 8

            df['volatility'] = df['log_ret'].rolling(
                window=window).std() * np.sqrt(
                window)

            _volatility = df.volatility.tail(1)
            _volatility.index = [df.name]
            _change = pd.concat([df['price'].head(1), df['price'].tail(1)])
            _change = _change.pct_change().rename('change') \
                .dropna(how='any')
            _change.index = [df.name]

            _df = pd.concat((_volatility, _change), axis=1)
            volatility_change.append(_df)
        df = pd.concat(volatility_change)
        df.index.rename('pair', inplace=True)
        df.reset_index(drop=False, inplace=True)
        df['partial_sum'] = df.volatility * df.change
        out = df.groupby('pair').partial_sum.agg('sum')
        out.rename('volatility_change', inplace=True)
        df = out.to_frame()
        df.reset_index(drop=False, inplace=True)
        df.set_index('volatility_change', inplace=True)
        df.sort_index(inplace=True, ascending=False)

        # df = df[df.index > 0.001]

        alog.info(df)
        return df

    @property
    def pairs(self):
        query = f'show measurements;'
        pair_frames = []
        for m in self.query(query).get_points('measurements'):
            name = m['name']
            if name.endswith('_40') and f'{self.filter}_' in name:
                pair_frames.append(name)

        return pair_frames

    def get_frames(self):
        pairs = self.query_pairs()
        series = [series['name'] for series in pairs._get_series()]

        frames = []

        for s in series:
            symbol = re.match(r'^([A-Z]+)', s).group(0)
            df = DataFrame([m for m in pairs.get_points(s)])
            df.name = symbol
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            frames.append(df)

        return frames

    def query_pairs(self):
        now = self.now()
        start_date = now - self.interval_delta
        end_date = now

        start_date = self.format_date_query(start_date)
        end_date = self.format_date_query(end_date)

        channels = ', '.join(self.pairs)

        query = f'SELECT first(best_ask) AS price FROM {channels} ' \
                f'WHERE time >= {start_date} AND time <= {end_date} GROUP BY ' \
                f'time({self.group_by});'

        alog.info(query)

        return self.query(query)

    # def convert(
    #     window_size,
    #     group_by,
    #     **kwargs
    # ):
    # frame_for_symbol(group_by, kwargs, window_size)


class VolatilityChangeEmitter(VolatilityChange, Messenger):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)



@click.command()
@click.option('--interval', '-i', default='1d', type=str)
@click.option('--filter', '-f', default='BNB', type=str)
@click.option('--database_name', '-d', default='binance', type=str)
@click.option('--window-size', '-w', default='15m', type=str)
@click.option('--group-by', '-g', default='30m', type=str)
@click.option('--plot', '-p', is_flag=True)
def main(**kwargs):
    VolatilityChangeEmitter(**kwargs)


if __name__ == '__main__':
    main()
