#!/usr/bin/env python
from collections import deque
from datetime import timedelta

from baselines.a2c.prediction_emitter import FrameNormalizer
from exchange_data.emitters import Messenger
from exchange_data.emitters.binance.volatility_change_emitter import \
    MeasurementFrame
from pandas import DataFrame
from pytimeparse.timeparse import timeparse
import alog
import click
import json
import numpy as np
import pandas
import pandas as pd
import re

from exchange_data.emitters.trading_window_emitter import TradingWindowEmitter
from exchange_data.streamers._orderbook_level import OrderBookLevelStreamer

pd.options.plotting.backend = 'plotly'


class OrderBookFrame(MeasurementFrame, FrameNormalizer):
    volume_max = 0.0

    def __init__(
        self,
        depth,
        database_name,
        max_volume_quantile,
        interval,
        sequence_length,
        window_size,
        symbol,
        volatility_intervals=False,
        **kwargs
    ):
        super().__init__(
            batch_size=1,
            database_name=database_name,
            interval=interval,
            symbol=symbol,
            **kwargs)
        self.max_volume_quantile = max_volume_quantile
        self.window_size = window_size
        self.depth = depth
        self.symbol = symbol
        self.volatility_intervals = volatility_intervals
        self.sequence_length = sequence_length

    def frame(self):
        frames = []

        for interval in self.intervals:
            self.start_date = interval[0]
            self.end_date = interval[1]
            frames.append(self.load_frames())

        df = pd.concat(frames)

        df = df.set_index('time')
        df = df.sort_index()
        df.dropna(how='any', inplace=True)

        alog.info(df)

        imgs = []

        orderbook_img = df.orderbook_img.to_numpy().tolist()
        df.drop(['orderbook_img'], axis=1)
        orderbook_img = np.asarray(orderbook_img)

        alog.info(orderbook_img.shape)

        orderbook_img = np.concatenate((
            orderbook_img[:, :, 0],
            orderbook_img[:, :, 1]),
            axis=2
        )

        orderbook_img = np.sort(orderbook_img, axis=2)

        alog.info(orderbook_img.shape)

        orderbook_img = np.delete(orderbook_img, 0, axis=3)

        self.quantile = np.quantile(orderbook_img, self.max_volume_quantile)

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
                                           group_by='2m',
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
@click.option('--interval', '-i', default='4h', type=str)
@click.option('--plot', '-p', is_flag=True)
@click.option('--sequence-length', '-l', default=48, type=int)
@click.option('--tick', is_flag=True)
@click.option('--max-volume-quantile', '-m', default=0.99, type=float)
@click.option('--volatility-intervals', '-v', is_flag=True)
@click.option('--window-size', '-w', default='3m', type=str)
@click.argument('symbol', type=str)
def main(**kwargs):
    frame = OrderBookFrame(**kwargs).frame()


if __name__ == '__main__':
    main()
