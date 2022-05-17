#!/usr/bin/env python

from collections import deque
from datetime import timedelta
from exchange_data.data.measurement_frame import MeasurementFrame
from exchange_data.streamers._orderbook_level import OrderBookLevelStreamer
from pandas import DataFrame
from pathlib import Path
from pytimeparse.timeparse import timeparse

import alog
import click
import json
import numpy as np
import pandas as pd

from exchange_data.utils import DateTimeUtils

pd.options.plotting.backend = 'plotly'


class OrderBookFrameDirectoryInfo(object):
    def __init__(
        self,
        directory_name=None,
        directory=None,
        **kwargs
    ):
        if directory_name is None:
            raise Exception()

        if directory is None:
            directory = \
                f'{Path.home()}/.exchange-data/orderbook_frame/{directory_name}'

        self.directory_name = directory_name
        self.directory = Path(directory)

        if not self.directory.exists():
            self.directory.mkdir(parents=True)

        super().__init__(**kwargs)


class OrderBookFrame(OrderBookFrameDirectoryInfo, MeasurementFrame):
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
        offset_interval='0h',
        round_decimals=4,
        max_volume_quantile=0.99,
        quantile=0.0,
        trade_volume_max=0.0,
        cache=False,
        **kwargs
    ):
        super().__init__(
            directory_name=symbol,
            database_name=database_name,
            interval=interval,
            symbol=symbol,
            depth=depth,
            sequence_length=sequence_length,
            **kwargs)

        self.trade_volume_max = trade_volume_max
        self.offset_interval = offset_interval
        self.group_by_delta = pd.Timedelta(seconds=timeparse(self.group_by))
        self.max_volume_quantile = max_volume_quantile
        self.quantile = quantile
        self.round_decimals = round_decimals
        self.window_size = window_size
        self.depth = 0
        self.output_depth = depth
        self.symbol = symbol
        self.sequence_length = sequence_length
        self.cache = cache
        self.filename = Path(self.directory / f'{interval}_{offset_interval}.pickle')

    def _frame(self):
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

        return df

    def trade_volume_query(self):
        start_date = self.start_date
        end_date = self.end_date

        start_date = self.format_date_query(start_date)
        end_date = self.format_date_query(end_date)
        channel_name = f'{self.symbol}_trade'

        query = f'SELECT sum(quantity) AS volume FROM {channel_name} ' \
            f'WHERE time >= {start_date} AND time <= {end_date} GROUP BY ' \
                f'time({self.group_by});'

        trades = self.query(query)

        df = pd.DataFrame(columns=['time', 'volume'])

        for data in trades.get_points(channel_name):
            timestamp = DateTimeUtils.parse_db_timestamp(
                data['time'])
            data['time'] = timestamp
            df.loc[df.shape[0], :] = data

        return df

    @property
    def trade_volume_frame(self):
        frames = []

        for interval in self.intervals:
            self.start_date = interval[0]
            self.end_date = interval[1]
            frames.append(self.trade_volume_query())

        df = pd.concat(frames)

        df.drop_duplicates(subset=['time'], inplace=True)

        df = df.set_index('time')
        df = df.sort_index()
        df.dropna(how='any', inplace=True)

        return df

    @property
    def frame(self):
        if self.cache and self.filename.exists():
            df = pd.read_pickle(str(self.filename))
            self.quantile = df.attrs['quantile']

            alog.info(f'#### {self.quantile} ####')

            self.trade_volume_max = df.attrs['trade_volume_max']
            return df

        df = self._frame()

        orderbook_img = df.orderbook_img.to_numpy().tolist()
        df.drop(['orderbook_img'], axis=1)
        orderbook_img = np.asarray(orderbook_img)

        orderbook_img = np.concatenate((
            orderbook_img[:, :, 0],
            orderbook_img[:, :, 1]),
            axis=2
        )

        orderbook_img = np.absolute(orderbook_img)

        for frame_ix in range(orderbook_img.shape[0]):
            orderbook = orderbook_img[frame_ix]
            shape = orderbook.shape
            new_ob = np.zeros((shape[0], shape[1], 1))

            last_frame_price = orderbook[-1][:, 0]

            for i in range(shape[0]):
                frame = orderbook[i]

                # alog.info(np.squeeze(frame))

                for l in range(frame.shape[0]):
                    level = frame[l]

                    price, volume = level

                    last_frame_index = np.where(last_frame_price == price)

                    new_ob[i, last_frame_index[0], 0] = np.asarray([volume])

            orderbook_img[frame_ix] = new_ob

        orderbook_img = np.delete(orderbook_img, 1, axis=3)

        if self.quantile == 0.0:
            self.quantile = \
                np.quantile(orderbook_img.flatten(), self.max_volume_quantile)
            if self.quantile == 0.0:
                self.quantile = 1.0

        orderbook_img = orderbook_img / self.quantile

        orderbook_img = np.clip(orderbook_img, a_min=0.0, a_max=1.0)

        orderbook_img = self.add_trade_volume(orderbook_img)

        # alog.info(alog.pformat(orderbook_img[0].tolist()))
        # raise Exception()

        df['orderbook_img'] = [
            orderbook_img[i] for i in range(0, orderbook_img.shape[0])
        ]

        df.attrs['trade_volume_max'] = self.trade_volume_max
        df.attrs['quantile'] = self.quantile

        self.cache_frame(df)

        return df

    def add_trade_volume(self, orderbook_img):
        # add trade volume
        trades = self.trade_volume_frame.reset_index(drop=True)
        new_shape = list(orderbook_img.shape)
        new_shape[2] = new_shape[2] + 1
        orderbook_img = np.resize(orderbook_img, new_shape)
        trades = np.squeeze(trades.to_numpy())
        if self.trade_volume_max == 0.0:
            self.trade_volume_max = np.quantile(trades, 1.0)
        trades = trades / self.trade_volume_max

        for index in range(len(trades)):
            if index <= len(orderbook_img) - 1:
                ord_img = orderbook_img[index]

                frame_size = len(ord_img)

                for i in range(frame_size):
                    trade_index = index - frame_size + i
                    ord_img[i][-1] = [trades[trade_index]]

                # alog.info(np.squeeze(ord_img))

        return orderbook_img

    @property
    def intervals(self):
        self.reset_interval()
        offset_interval = timedelta(seconds=timeparse(self.offset_interval))
        start_date = self.start_date - offset_interval
        end_date = self.end_date - offset_interval

        return [(start_date, end_date)]

    def load_frames(self):
        frames = []

        self.start_date = self.start_date - timedelta(seconds=(timeparse(
            self.group_by) * self.sequence_length))

        levels = OrderBookLevelStreamer(
            database_name=self.database_name,
            depth=self.depth,
            end_date=self.end_date,
            group_by=self.group_by,
            start_date=self.start_date,
            symbol=self.symbol,
            window_size=self.window_size
        )

        df = pd.DataFrame(
            columns=['timestamp', 'best_ask', 'best_bid', 'orderbook_img'])

        for row in levels:
            df.loc[df.shape[0], :] = row

        orderbook_imgs = deque(maxlen=self.sequence_length)

        max_shape = (2, self.output_depth, 2)

        for ix, timestamp, best_ask, best_bid, orderbook_img in df.itertuples():
            if orderbook_img is not None:
                orderbook_img = np.asarray(json.loads(orderbook_img))

                orderbook_img[0][0] = \
                    orderbook_img[0][0].round(self.round_decimals)
                orderbook_img[1][0] = \
                    orderbook_img[1][0].round(self.round_decimals)

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
                    orderbook_img[1, :right_len, :2] = right[:right_len, :2]
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

    def cache_frame(self, df):
        df.to_pickle(str(self.filename))

    def group_price_levels(self, orderbook_side):
        groups = dict()

        for price_vol in orderbook_side.tolist():
            price = price_vol[0]
            vol = groups.get(price, 0.0)
            vol += price_vol[1]
            groups[price] = vol

        result = np.array([[p, v] for p, v in groups.items()])

        return result


@click.command()
@click.option('--database_name', '-d', default='binance', type=str)
@click.option('--depth', default=72, type=int)
@click.option('--group-by', '-g', default='30s', type=str)
@click.option('--interval', '-i', default='10m', type=str)
@click.option('--offset-interval', '-o', default='3h', type=str)
@click.option('--plot', '-p', is_flag=True)
@click.option('--sequence-length', '-l', default=48, type=int)
@click.option('--round-decimals', '-D', default=4, type=int)
@click.option('--tick', is_flag=True)
@click.option('--cache', is_flag=True)
@click.option('--max-volume-quantile', '-m', default=0.99, type=float)
@click.option('--window-size', '-w', default='3m', type=str)
@click.argument('symbol', type=str)
def main(**kwargs):
    df = OrderBookFrame(**kwargs).frame

    pd.set_option('display.max_rows', len(df) + 1)

    alog.info(df)

    obook = df.orderbook_img.to_numpy()

    obook = np.squeeze(obook[-1])

    alog.info(obook)


if __name__ == '__main__':
    main()
