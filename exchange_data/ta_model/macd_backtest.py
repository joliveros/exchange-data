#!/usr/bin/env python

import click
import pandas as pd
from pandas import DataFrame

from exchange_data.ta_model.backtest import BackTest
from exchange_data.trading import Positions


class MacdBackTest(BackTest):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

    def label_position(
        self,
        df=None,
        span_1=12,
        span_2=26,
        span_3=9,
        **kwargs
    ):
        if df is None:
            df = self.ohlc.copy()

        df.reset_index(drop=False, inplace=True)
        df_close = df['close']

        adjust = True

        exp1 = df_close.ewm(span=span_1, adjust=adjust).mean()

        exp2 = df_close.ewm(span=span_2, adjust=adjust).mean()

        macd = exp1 - exp2

        exp3 = macd.ewm(span=span_3, adjust=adjust).mean()

        minDf = DataFrame(exp3)

        minDf['time'] = df['time']
        minDf.columns = ['avg', 'time']
        minDf = minDf.set_index('time')
        maxDf = minDf.copy()
        minDf['min'] = \
            minDf.avg[(minDf.avg.shift(1) > minDf.avg) & (
                minDf.avg.shift(-1) > minDf.avg)]
        maxDf['max'] = \
            maxDf.avg[(maxDf.avg.shift(1) < maxDf.avg) & (
                maxDf.avg.shift(-1) < maxDf.avg)]
        maxDf = maxDf.reset_index(drop=False)
        maxDf = maxDf.dropna()
        maxDf = maxDf.drop(columns=['max'])
        maxDf['type'] = 'max'
        minDf = minDf.reset_index(drop=False)
        minDf = minDf.dropna()
        minDf = minDf.drop(columns=['min'])
        minDf['type'] = 'min'
        minmax_pairs = \
            sorted(
                tuple(zip(maxDf.time, maxDf.avg, maxDf.type)) + tuple(
                    zip(minDf.time, minDf.avg, minDf.type))
            )

        df = self.frame.copy()
        df['position'] = Positions.Flat

        for d, val, type in minmax_pairs:
            position = Positions.Flat

            if type == 'min':
                position = Positions.Long

            df.loc[pd.DatetimeIndex(df.index) > d, 'position'] = \
                position

        # pd.set_option('display.max_rows', len(df) + 1)

        df.dropna(how='any', inplace=True)

        return df


@click.command()
@click.option('--database-name', '-d', default='binance', type=str)
@click.option('--group-by', '-g', default='1m', type=str)
@click.option('--group-by-min', '-m', default='4', type=str)
@click.option('--interval', '-i', default='12h', type=str)
@click.option('--plot', '-p', is_flag=True)
@click.option('--window-size', '-w', default='2h', type=str)
@click.option('--session-limit', '-s', default=100, type=int)
@click.argument('symbol', type=str)
def main(**kwargs):
    MacdBackTest(**kwargs).test()


if __name__ == '__main__':
    main()
