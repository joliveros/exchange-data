#!/usr/bin/env python

from exchange_data.data.orderbook_frame import OrderBookFrame
from exchange_data.emitters.backtest_base import BackTestBase
from exchange_data.trading import Positions

import alog
import click
import pandas as pd

pd.options.plotting.backend = 'plotly'


class MaxMinFrame(OrderBookFrame, BackTestBase):
    def __init__(self, symbol, **kwargs):
        super().__init__(symbol=symbol, **kwargs)
        BackTestBase.__init__(self, symbol=symbol, **kwargs)

    def label_position(self):
        df = self.ohlc.copy()

        df.reset_index(drop=False, inplace=True)

        alog.info(df)

        df['min'] = \
            df.close[(df.close.shift(1) > df.close) & (
                df.close.shift(-1) > df.close)]
        df['max'] = \
            df.close[(df.close.shift(1) < df.close) & (
                df.close.shift(-1) < df.close)]

        df['position'] = Positions.Flat

        min = list(df[['time', 'min']].dropna(how='any').itertuples())
        min = [(r.time, 'min') for r in min]

        max = list(df[['time', 'max']].dropna(how='any').itertuples())
        max = [(r.time, 'max') for r in max]

        minmax_pairs = sorted(min + max)

        df = df.set_index('time')

        for d, val in minmax_pairs:

            if val == 'min':
                position = Positions.Long
            else:
                position = Positions.Flat

            df.loc[pd.DatetimeIndex(df.index) >= d, 'position'] = \
                position

        return df

    def label_positive_change(
        self,
        **kwargs
    ):
        position = self.label_position().position

        df = self.frame.copy()

        df['expected_position'] = position

        df['expected_position'] = df.expected_position.ffill()

        df.dropna(how='any', inplace=True)

        alog.info(f'### volume_max {self.quantile} ####')

        return df


@click.command()
@click.option('--database_name', '-d', default='binance', type=str)
@click.option('--depth', default=40, type=int)
@click.option('--group-by', '-g', default='1m', type=str)
@click.option('--group-by-min', '-G', default=5, type=int)
@click.option('--interval', '-i', default='3h', type=str)
@click.option('--max-volume-quantile', '-m', default=0.99, type=float)
@click.option('--plot', '-p', is_flag=True)
@click.option('--sequence-length', '-l', default=48, type=int)
@click.option('--tick', is_flag=True)
@click.option('--volatility-intervals', '-v', is_flag=True)
@click.option('--window-size', '-w', default='3m', type=str)
@click.argument('symbol', type=str)
def main(**kwargs):
    df = MaxMinFrame(**kwargs).label_positive_change()

    # pd.set_option('display.max_rows', len(df) + 1)

    alog.info(df)



if __name__ == '__main__':
    main()