#!/usr/bin/env python

from exchange_data import Database, Measurement
from exchange_data.data.price_frame import PriceFrame
from exchange_data.emitters.backtest_base import BackTestBase
from exchange_data.trading import Positions
from optuna import create_study, Trial
from pandas import DataFrame

import alog
import click
import json
import pandas as pd



class BackTest(PriceFrame, BackTestBase):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        BackTestBase.__init__(self, **kwargs)

    def label_position(self, short_period=8, long_period=22, group_by_min=None):
        if group_by_min:
            self.group_by_min = group_by_min

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
        self.df = df

    def test(self, **kwargs):
        self.label_position(**kwargs)
        self.capital = 1.0
        df = self.df.copy()
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


class TuneMACDSignal(BackTest, Database):
    def __init__(self, session_limit=100, **kwargs):
        super().__init__(**kwargs)

        self.study = create_study(
            study_name=self.symbol, direction='maximize')

        self.study.optimize(self.run, n_trials=session_limit)

        alog.info(alog.pformat(vars(self.study.best_trial)))

        data = dict(
            params=json.dumps(self.study.best_trial.params),
            symbol=self.symbol,
            value=self.study.best_trial.value,
        )

        m = Measurement(fields=data, measurement='macd_params')

        self.write_points([m.__dict__])


    def run(self, trial: Trial):
        params = dict(
            short_period = trial.suggest_int('short_period', 1, 36),
            long_period = trial.suggest_int('long_period', 1, 36),
            group_by_min = trial.suggest_int('group_by_min', 1, 30)
        )

        return self.test(**params)


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
    TuneMACDSignal(**kwargs)


if __name__ == '__main__':
    main()
