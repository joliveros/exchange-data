#!/usr/bin/env python

from datetime import timedelta
from exchange_data.data.measurement_frame import MeasurementFrame
from exchange_data.emitters import Messenger
from pandas import DataFrame
from pytimeparse.timeparse import timeparse

import alog
import click
import numpy as np
import pandas as pd
import re

pd.options.plotting.backend = 'plotly'


class VolatilityChange(MeasurementFrame):
    def __init__(self, filter, pairs_limit=0, **kwargs):
        super().__init__(
            batch_size=1,
            **kwargs)
        now = self.now()
        self.start_date = now - self.interval
        self.end_date = now
        self.pairs_limit = pairs_limit
        self.filter = filter

    def tick(self):
        self.append(self.volatility_change())

    def last_frame(self):
        last_row = self.frame().iloc[-1]
        df = last_row.to_frame()
        df.reset_index(inplace=True, drop=False)
        df.columns = ['pair', self.name]
        df.set_index(self.name, inplace=True)
        df.sort_index(inplace=True, ascending=False)
        alog.info(df)
        return df

    def volatility_change(self):
        self.frames = self.get_frames()
        # alog.info(alog.pformat(self.frames))
        volatility_change = []
        for df in self.frames:
            df['log_ret'] = np.log(df['price'] / df['price'].shift(1))
            df.dropna(how='any', inplace=True)

            df['volatility'] = df['price'].rolling(18).std(ddof=0)

            df['volume_avg'] = df.volume.rolling(18).std(ddof=0)

            _volume_avg = df.volume_avg.tail(1)
            _volume_avg.index = [df.name]

            _volatility = df.volatility.tail(1)
            _volatility.index = [df.name]
            _change = pd.concat([df['price'].head(1), df['price'].tail(1)])

            # alog.info(_change)

            _change = _change.pct_change().rename('change') \
                .dropna(how='any')
            _change.index = [df.name]

            _df = pd.concat((_volatility, _change, _volume_avg), axis=1)

            volatility_change.append(_df)

        df = pd.concat(volatility_change)
        df.index.rename('pair', inplace=True)
        df.reset_index(drop=False, inplace=True)
        df['partial_sum'] = df.volatility * df.change * df.volume_avg
        out = df.groupby('pair').partial_sum.agg('sum')
        out.rename('volatility_change', inplace=True)
        df = out.to_frame()
        df.reset_index(drop=False, inplace=True)
        df.set_index('volatility_change', inplace=True)
        df.sort_index(inplace=True, ascending=False)
        df.reset_index(drop=False, inplace=True)

        # df = df[df.index > 0.001]

        pd.set_option('display.max_rows', len(df) + 1)

        alog.info(df)

        self.df = df
        return df

    @property
    def pairs(self):
        query = f'show measurements;'
        pair_frames = []
        for m in self.query(query).get_points('measurements'):
            name = m['name']
            if name.endswith('_40') and f'{self.filter}_' in name:
                pair_frames.append(name)

        if self.pairs_limit > 0:
            return pair_frames[:self.pairs_limit]

        return pair_frames

    def get_volume(self):
        volume = self.query_volume()
        series = volume._get_series()

        volume_dfs = []

        for s in series:
            symbol = s['tags']['symbol']
            values = np.array(s['values'])

            df = DataFrame(dict(
                time=values[:, 0] * 1000000,
                volume=values[:,1],
                price=values[:, 2]
            ))

            df.volume = pd.to_numeric(df.volume)
            df.price = pd.to_numeric(df.price)
            df.volume = df.volume / df.price
            df.time = pd.to_datetime(df.time)

            df.fillna(0.0, inplace=True)

            df.name = symbol

            volume_dfs.append(df)

        return volume_dfs

    def get_frames(self):
        volume_dfs = self.get_volume()

        pairs = self.query_pairs()

        series = [series['name'] for series in pairs._get_series()]

        frames = []

        for s in series:
            symbol = re.match(r'^([A-Z]+)', s).group(0)
            vdf = [df for df in volume_dfs if df.name == symbol][0]
            df = DataFrame([m for m in pairs.get_points(s)])
            df['volume'] = vdf.volume
            df.name = symbol
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            frames.append(df)

        return frames

    def query_pairs(self):
        channels = ', '.join(self.pairs)

        query = f'SELECT first(best_ask) AS price FROM {channels} ' \
                f'WHERE time >= {self.formatted_start_date} AND time <= ' \
                f'{self.formatted_end_date} GROUP BY ' \
                f'time({self.group_by});'

        alog.info(query)

        return self.query(query)

    def query_volume(self):
        query = f'SELECT SUM("quantity"), FIRST("price") FROM "trade" ' \
                f'WHERE time >= {self.formatted_start_date} AND time <= ' \
                f'{self.formatted_end_date} GROUP BY ' \
                f'time({self.group_by}),"symbol";'

        alog.info(query)

        return self.query(query)


class VolatilityChangeEmitter(VolatilityChange, Messenger):
    def __init__(self, plot=False, tick=False, **kwargs):
        super().__init__(plot=plot, **kwargs)

        if tick:
            self.tick()

    def plot(self):
        self.df.plot().show()

    def frame(self):
        query = f'SELECT first(*) AS data FROM {self.name} WHERE time >=' \
                f' {self.formatted_start_date} AND ' \
                f'time <= {self.formatted_end_date} GROUP BY time(' \
                f'{self.group_by})'

        alog.info(query)

        frames = []

        for data in self.query(query).get_points(self.name):
            data = data['data_data'] or {}

            if type(data) is str:
                data = pd.read_json(data)
                frames.append(data)

        return frames

@click.command()
@click.option('--interval', '-i', default='12h', type=str)
@click.option('--filter', '-f', default='BNB', type=str)
@click.option('--database_name', '-d', default='binance', type=str)
@click.option('--window-size', '-w', default='15m', type=str)
@click.option('--group-by', '-g', default='15m', type=str)
@click.option('--pairs-limit', '-l', default=0, type=int)
@click.option('--plot', '-p', is_flag=True)
@click.option('--tick', is_flag=True)
def main(**kwargs):
    emitter = VolatilityChangeEmitter(**kwargs)


if __name__ == '__main__':
    main()
