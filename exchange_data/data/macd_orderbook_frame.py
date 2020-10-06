#!/usr/bin/env python
from redlock import RedLock, RedLockError

from cached_property import cached_property, cached_property_with_ttl

from exchange_data import settings
from exchange_data.data.labeled_orderbook_frame import LabeledOrderBookFrame
from pytimeparse.timeparse import timeparse

import alog
import click
import pandas as pd
import time as t

from exchange_data.ta_model.tune_macd import MacdParamFrame
from exchange_data.ta_model.tune_macd_single_pass import \
    TuneMACDSignalSinglePass

pd.options.plotting.backend = 'plotly'


class MacdOrderBookFrame(LabeledOrderBookFrame):

    @cached_property
    def macd_params(self):
        if not self._macd_params.empty:
            return self._macd_params
        else:
            return self.gen_macd_params()

    @cached_property_with_ttl(ttl=2)
    def _macd_params(self):
        return MacdParamFrame(database_name=self.database_name).frame()

    def gen_macd_params(self):
        try:
            return self._gen_macd_params()
        except RedLockError:
            t.sleep(2)

            if not self._macd_params.empty:
                return self._macd_params
            else:
                return self.gen_macd_params()

    def _gen_macd_params(self):
        # with self.macd_params_lock():
        result = TuneMACDSignalSinglePass(
            n_jobs=1,
            group_by_min=1,
            **self._kwargs
        ).run_study()
        return result

    def macd_params_lock(self):
        lock_name = f'macd_params_lock_{self.symbol}'

        alog.info(lock_name)

        return RedLock(lock_name, [dict(
            host=settings.REDIS_HOST,
            db=0
        )], retry_delay=200, retry_times=3, ttl=timeparse('3m') * 1000)

    def label_positive_change(
        self,
        **kwargs
    ):
        position = self.single_pass_backtest.label_position(
            **self.macd_params).position

        df = self.frame.copy()

        df['expected_position'] = position

        df.dropna(how='any', inplace=True)

        alog.info(f'### volume_max {self.quantile} ####')

        return df


@click.command()
@click.option('--database_name', '-d', default='binance', type=str)
@click.option('--depth', default=40, type=int)
@click.option('--group-by', '-g', default='1m', type=str)
@click.option('--interval', '-i', default='3h', type=str)
@click.option('--max-volume-quantile', '-m', default=0.99, type=float)
@click.option('--plot', '-p', is_flag=True)
@click.option('--sequence-length', '-l', default=48, type=int)
@click.option('--session-limit', '-s', default=200, type=int)
@click.option('--tick', is_flag=True)
@click.option('--volatility-intervals', '-v', is_flag=True)
@click.option('--window-size', '-w', default='3m', type=str)
@click.argument('symbol', type=str)
def main(**kwargs):
    df = MacdOrderBookFrame(**kwargs).label_positive_change()

    # pd.set_option('display.max_rows', len(df) + 1)

    alog.info(df)



if __name__ == '__main__':
    main()
