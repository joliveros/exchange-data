#!/usr/bin/env python
from redlock import RedLock, RedLockError

from baselines.a2c.prediction_emitter import FrameNormalizer
from cached_property import cached_property, cached_property_with_ttl
from collections import deque
from datetime import timedelta

from exchange_data import settings
from exchange_data.data.measurement_frame import MeasurementFrame
from exchange_data.data.orderbook_frame import OrderBookFrame
from exchange_data.emitters import Messenger
from exchange_data.emitters.trading_window_emitter import TradingWindowEmitter
from exchange_data.streamers._orderbook_level import OrderBookLevelStreamer
from pandas import DataFrame
from pytimeparse.timeparse import timeparse

import alog
import click
import json
import numpy as np
import pandas as pd
import time as t

from exchange_data.ta_model.single_pass_backtest import SinglePassBackTest
from exchange_data.ta_model.tune_macd import MacdParamFrame
from exchange_data.ta_model.tune_macd_single_pass import \
    TuneMACDSignalSinglePass

pd.options.plotting.backend = 'plotly'


class MacdOrderBookFrame(OrderBookFrame):

    def __init__(
        self,
        **kwargs
    ):
        self._kwargs = kwargs
        super().__init__(
            **kwargs)

        self.single_pass_backtest = SinglePassBackTest(**self._kwargs)

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

    @property
    def frame(self):
        frames = []

        for interval in self.intervals:
            self.start_date = interval[0]
            self.end_date = interval[1]
            frames.append(self.load_frames())

        df = pd.concat(frames)

        alog.info(df)

        df.drop_duplicates(subset=['time'], inplace=True)

        df = df.set_index('time')
        df = df.sort_index()
        df.dropna(how='any', inplace=True)

        alog.info(df)

        imgs = []

        orderbook_img = df.orderbook_img.to_numpy().tolist()
        df.drop(['orderbook_img'], axis=1)
        orderbook_img = np.asarray(orderbook_img)

        orderbook_img = np.concatenate((
            orderbook_img[:, :, 0],
            orderbook_img[:, :, 1]),
            axis=2
        )

        orderbook_img = np.sort(orderbook_img, axis=2)

        orderbook_img = np.delete(orderbook_img, 0, axis=3)

        if self.quantile == 0.0:
            self.quantile = np.quantile(orderbook_img,
                                         self.max_volume_quantile)

        orderbook_img = orderbook_img / self.quantile

        orderbook_img = np.clip(orderbook_img, a_min=0.0, a_max=1.0)

        df['orderbook_img'] = [
            orderbook_img[i] for i in range(0, orderbook_img.shape[0])
        ]

        alog.info(df)

        return df

    @property
    def intervals(self):
        if self.volatility_intervals:
            twindow = TradingWindowEmitter(interval=self.interval_str,
                                           group_by='1m',
                                           database_name=self.database_name,
                                           plot=False,
                                           symbol=self.symbol)
            twindow.next_intervals()

            return twindow.intervals
        else:
            return [(self.start_date, self.end_date)]

    def load_frames(self):
        frames = []

        self.start_date = self.start_date - timedelta(seconds=timeparse(
            self.group_by) * self.sequence_length * 2)

        levels = OrderBookLevelStreamer(
            database_name=self.database_name,
            depth=self.depth,
            end_date=self.end_date,
            groupby=self.group_by,
            start_date=self.start_date,
            symbol=self.symbol,
            window_size=self.window_size
        )

        orderbook_imgs = deque(maxlen=self.sequence_length)

        for timestamp, best_ask, best_bid, orderbook_img in levels:

            if orderbook_img is not None:
                orderbook_img = np.asarray(json.loads(orderbook_img))
                orderbook_imgs.append(orderbook_img)

                if len(orderbook_imgs) == self.sequence_length:
                    frame = dict(
                        time=timestamp,
                        best_ask=best_ask,
                        best_bid=best_bid,
                        orderbook_img=np.asarray(list(orderbook_imgs.copy()))
                    )

                    frames.append(frame)

        df = DataFrame(frames)

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