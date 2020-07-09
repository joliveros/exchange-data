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
    def __init__(
        self,
        depth,
        database_name,
        interval,
        sequence_length,
        symbol,
        volume_max,
        volatility_intervals=False,
        **kwargs
    ):
        super().__init__(
            batch_size=1,
            database_name=database_name,
            interval=interval,
            symbol=symbol,
            **kwargs)
        self.depth = depth
        self.symbol = symbol
        self.volatility_intervals = volatility_intervals
        self.volume_max = volume_max
        self.sequence_length = sequence_length

    def frame(self):
        frames = []
        for interval in self.intervals:
            self.start_date = interval[0]
            self.end_date = interval[1]
            df = self.load_frames()

            frames.append(df)
        df = pd.concat(frames)
        alog.info(f'### observed_max {self.observed_max} ###')
        df = df.set_index('time')
        df = df.sort_index()

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
            self.group_by) * self.sequence_length)

        levels = OrderBookLevelStreamer(
            database_name=self.database_name,
            depth=self.depth,
            end_date=self.end_date,
            groupby=self.group_by,
            start_date=self.start_date,
            symbol=self.symbol,
            window_size='15m'
        )

        orderbook_imgs = deque(maxlen=self.sequence_length)

        for timestamp, best_ask, best_bid, orderbook_img in levels:
            if orderbook_img:
                orderbook_img = np.asarray(json.loads(orderbook_img))
                orderbook_img = self.normalize_frame(orderbook_img, self.volume_max)
                orderbook_imgs.append(orderbook_img)

                if len(orderbook_imgs) == self.sequence_length:
                    frame = dict(
                        time=timestamp,
                        best_ask=best_ask,
                        best_bid=best_bid,
                        orderbook_img=np.array(list(orderbook_imgs.copy()))
                    )

                    frames.append(frame)

        return DataFrame(frames)


@click.command()
@click.option('--database_name', '-d', default='binance', type=str)
@click.option('--depth', default=40, type=int)
@click.option('--group-by', '-g', default='1m', type=str)
@click.option('--interval', '-i', default='3d', type=str)
@click.option('--plot', '-p', is_flag=True)
@click.option('--sequence-length', '-l', default=48, type=int)
@click.option('--tick', is_flag=True)
@click.option('--volatility-intervals', '-v', is_flag=True)
@click.option('--volume-max', default=12000, type=float)
@click.option('--window-size', '-w', default='15m', type=str)
@click.argument('symbol', type=str)
def main(**kwargs):
    frame = OrderBookFrame(**kwargs)


if __name__ == '__main__':
    main()
