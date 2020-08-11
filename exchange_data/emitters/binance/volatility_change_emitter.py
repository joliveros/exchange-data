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
    def __init__(self, filter, pairs_limit, **kwargs):
        super().__init__(
            batch_size=1,
            **kwargs)
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

            df['volatility'] = df['price'].rolling(20).std(ddof=0)

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
        start_date = now - self.interval
        end_date = now

        start_date = self.format_date_query(start_date)
        end_date = self.format_date_query(end_date)

        channels = ', '.join(self.pairs)

        query = f'SELECT first(best_ask) AS price FROM {channels} ' \
                f'WHERE time >= {start_date} AND time <= {end_date} GROUP BY ' \
                f'time({self.group_by});'

        alog.info(query)

        return self.query(query)


class VolatilityChangeEmitter(VolatilityChange, Messenger):
    def __init__(self, plot, interval, group_by, tick=False, **kwargs):
        alog.info(alog.pformat(kwargs))
        super().__init__(plot=plot, group_by='30m', interval='1d', **kwargs)

        if tick:
            self.tick()

        self.group_by = group_by
        self.interval = timedelta(seconds=timeparse(interval))

        plot = True

    def plot(self):
        self.df.plot().show()


@click.command()
@click.option('--interval', '-i', default='14m', type=str)
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
