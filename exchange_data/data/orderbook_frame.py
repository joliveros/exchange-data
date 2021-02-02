#!/usr/bin/env python

from baselines.a2c.prediction_emitter import FrameNormalizer
from cached_property import cached_property
from collections import deque
from datetime import timedelta
from exchange_data.data.measurement_frame import MeasurementFrame
from exchange_data.emitters.trading_window_emitter import TradingWindowEmitter
from exchange_data.streamers._orderbook_level import OrderBookLevelStreamer
from pandas import DataFrame
from pytimeparse.timeparse import timeparse

import alog
import click
import json
import numpy as np
import pandas as pd

pd.options.plotting.backend = 'plotly'


class OrderBookFrame(MeasurementFrame, FrameNormalizer):
    positive_change_count = 0
    min_consecutive_count = 1

    def __init__(
        self,
        database_name,
        depth,
        interval,
        sequence_length,
        symbol,
        window_size,
        offset_interval=None,
        round_decimals=4,
        max_volume_quantile=0.99,
        quantile=0.0,
        volatility_intervals=False,
        **kwargs
    ):
        super().__init__(
            batch_size=1,
            database_name=database_name,
            interval=interval,
            symbol=symbol,
            depth=0,
            **kwargs)

        self.offset_interval = offset_interval
        self.group_by_delta = pd.Timedelta(seconds=timeparse(self.group_by))
        self.max_volume_quantile = max_volume_quantile
        self.quantile = quantile
        self.round_decimals = round_decimals
        self.window_size = window_size
        self.depth = 0
        self.output_depth = depth
        self.symbol = symbol
        self.volatility_intervals = volatility_intervals
        self.sequence_length = sequence_length

    @cached_property
    def frame(self):
        frames = []

        for interval in self.intervals:
            self.start_date = interval[0]
            self.end_date = interval[1]
            frames.append(self.load_frames())

        df = pd.concat(frames)

        df.drop_duplicates(subset=['time'], inplace=True)

        df = df.set_index('time')
        df = df.sort_index()
        df.dropna(how='any', inplace=True)

        orderbook_img = df.orderbook_img.to_numpy().tolist()

        df.drop(['orderbook_img'], axis=1)

        orderbook_img = np.asarray(orderbook_img)

        orderbook_img = np.sort(orderbook_img, axis=3)

        orderbook_img = np.concatenate((
            orderbook_img[:, :, 0],
            orderbook_img[:, :, 1]),
            axis=2
        )

        orderbook_img = np.delete(orderbook_img, 0, axis=3)
        orderbook_img = np.absolute(orderbook_img)

        if self.quantile == 0.0:
            self.quantile = np.quantile(orderbook_img,
                                         self.max_volume_quantile)

        orderbook_img = orderbook_img / self.quantile

        orderbook_img = np.clip(orderbook_img, a_min=0.0, a_max=1.0)

        df['orderbook_img'] = [
            orderbook_img[i] for i in range(0, orderbook_img.shape[0])
        ]

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
            if self.offset_interval:
                offset_interval = timedelta(seconds=timeparse(self.offset_interval))
                self.start_date = self.start_date - offset_interval
                self.end_date = self.end_date - offset_interval

            return [(self.start_date, self.end_date)]

    def load_frames(self):
        frames = []

        self.start_date = self.start_date - timedelta(seconds=timeparse(
            self.group_by) * self.sequence_length)

        levels = OrderBookLevelStreamer(
            database_name=self.database_name,
            depth=self.depth,
            end_date=self.end_date,
            group_by=self.group_by,
            start_date=self.start_date,
            symbol=self.symbol,
            window_size=self.window_size
        )

        orderbook_imgs = deque(maxlen=self.sequence_length)

        max_shape = (2, self.output_depth, 2)

        for timestamp, best_ask, best_bid, orderbook_img in levels:

            if orderbook_img is not None:
                orderbook_img = np.asarray(json.loads(orderbook_img))
                orderbook_img[0][0] = orderbook_img[0][0].round(self.round_decimals)
                orderbook_img[1][0] = orderbook_img[1][0].round(self.round_decimals)

                left = orderbook_img[0].swapaxes(1, 0)
                right = orderbook_img[1].swapaxes(1, 0)

                left = np.sort(self.group_price_levels(left), axis=0)
                right = np.sort(self.group_price_levels(right), axis=0)
                right = np.flip(right, 0)

                orderbook_img = np.zeros(max_shape)

                left_len = left.shape[0]
                if left_len > self.output_depth:
                    left_len = self.output_depth

                right_len = right.shape[0]
                if right_len > self.output_depth:
                    right_len = self.output_depth

                if left.shape != (0,) and right.shape != (0,):
                    orderbook_img[0, :left_len, :2] = left[:left_len, :2]
                    orderbook_img[1, :right_len, :2 ] = right[:right_len, :2]
                    orderbook_imgs.append(orderbook_img)

                    if len(orderbook_imgs) == self.sequence_length:
                        _orderbook_imgs = np.asarray(list(
                            orderbook_imgs.copy()))

                        _orderbook_imgs[_orderbook_imgs == np.inf] = 0.0
                        _orderbook_imgs[_orderbook_imgs == -np.inf] = 0.0

                        frame = dict(
                            time=timestamp,
                            best_ask=best_ask,
                            best_bid=best_bid,
                            orderbook_img=_orderbook_imgs,
                            dtype=np.float16
                        )
                        frames.append(frame)

        df = DataFrame(frames)

        df = df.astype({"best_ask": np.float16, "best_bid": np.float16})

        return df

    def group_price_levels(self, orderbook_side):
        groups = dict()

        for price_vol in orderbook_side.tolist():
            price = price_vol[0]
            vol = groups.get(price, 0.0)
            vol += price_vol[1]
            groups[price] = vol

        return np.array([[p, v] for p, v in groups.items()])


@click.command()
@click.option('--database_name', '-d', default='binance', type=str)
@click.option('--depth', default=72, type=int)
@click.option('--group-by', '-g', default='1m', type=str)
@click.option('--interval', '-i', default='3h', type=str)
@click.option('--offset-interval', '-o', default='3h', type=str)
@click.option('--plot', '-p', is_flag=True)
@click.option('--sequence-length', '-l', default=48, type=int)
@click.option('--round-decimals', '-D', default=4, type=int)
@click.option('--tick', is_flag=True)
@click.option('--max-volume-quantile', '-m', default=0.99, type=float)
@click.option('--volatility-intervals', '-v', is_flag=True)
@click.option('--window-size', '-w', default='3m', type=str)
@click.argument('symbol', type=str)
def main(**kwargs):
    df = OrderBookFrame(**kwargs).frame

    # pd.set_option('display.max_rows', len(df) + 1)

    alog.info(alog.pformat(df.iloc[-1].orderbook_img[-1].tolist()))


if __name__ == '__main__':
    main()
