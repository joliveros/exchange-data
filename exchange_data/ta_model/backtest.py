#!/usr/bin/env python

import alog
import click
import pandas as pd
from pandas import DataFrame

from exchange_data.data.price_frame import PriceFrame
from exchange_data.emitters.backtest_base import BackTestBase
from exchange_data.trading import Positions


class BackTest(PriceFrame, BackTestBase):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        BackTestBase.__init__(self, **kwargs)

    def label_position(
        self,
        df=None,
        short_period=8,
        long_period=22,
        **kwargs
    ):
        if df is None:
            df = self.ohlc.copy()

        df.reset_index(drop=False, inplace=True)
        df_close = df['close']
        exp1 = df_close.ewm(span=short_period, adjust=False).mean()
        exp2 = df_close.ewm(span=long_period, adjust=False).mean()
        macd = exp1 - exp2
        exp3 = macd.ewm(span=9, adjust=False).mean()
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

    def test(self, **kwargs):
        df: DataFrame = self.frame.copy()

        for i in df.index.values:
            labeled_df = self.label_position(
                self.ohlc.truncate(after=i),
                **kwargs
            )

            position = labeled_df.iloc[-1].position

            df.loc[i, 'position'] = position

        self.capital = 1.0

        df.reset_index(drop=False, inplace=True)
        df['capital'] = self.capital
        df = df.apply(self.pnl, axis=1)
        pd.set_option('display.max_rows', len(df) + 1)
        alog.info(df)

        if self.capital > 50.0:
            return 0.0

        return self.capital

    def load_previous_frames(self, depth):
        pass


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
    BackTest(**kwargs).test()


if __name__ == '__main__':
    main()
