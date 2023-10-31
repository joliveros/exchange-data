#!/usr/bin/env python
from collections import deque
from datetime import timedelta
from exchange_data.data.measurement_frame import MeasurementFrame
from exchange_data.data.orderbook_frame_directory_info import (
    OrderBookFrameDirectoryInfo,
)
from exchange_data.streamers._orderbook_level import OrderBookLevelStreamer
from pandas import DataFrame
from pathlib import Path
from pytimeparse.timeparse import timeparse
from matplotlib import pyplot as plt
from exchange_data.utils import DateTimeUtils
from PIL import Image as im

import alog
import click
import cv2
import hashlib
import json
import numpy as np
import pandas as pd

# pd.options.plotting.backend = 'plotly'


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
        frame_width,
        offset_interval="0h",
        round_decimals=4,
        max_volume_quantile=0.99,
        quantile=0.0,
        trade_volume_max=0.0,
        change_max=0.0,
        cache=False,
        **kwargs,
    ):
        super().__init__(
            directory_name=symbol,
            database_name=database_name,
            interval=interval,
            symbol=symbol,
            depth=depth,
            sequence_length=sequence_length,
            **kwargs,
        )

        self.frame_width = frame_width
        self.change_max = change_max
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

        file_dict = dict(
            interval=interval,
            offset_interval=offset_interval,
            symbol=self.symbol,
            round_decimals=self.round_decimals,
            depth=self.depth,
            group_by=self.group_by,
            sequence_length=self.sequence_length,
        )

        hash = hashlib.sha256(bytes(json.dumps(file_dict), "utf8")).hexdigest()

        self.filename = Path(self.directory / f"{hash}.pickle")
        self._intervals = None

        self.kwargs = kwargs

    def _frame(self):
        frames = []

        for interval in self.intervals:
            self.start_date = interval[0]
            self.end_date = interval[1]
            # alog.info([
            #     str(self.start_date),
            #     str(self.end_date),
            #     str(self.end_date - self.start_date)
            # ])
            frames.append(self.load_frames())

        df = pd.concat(frames)

        df.drop_duplicates(subset=["time"], inplace=True)

        df = df.set_index("time")
        df = df.sort_index()
        df.dropna(how="any", inplace=True)

        self.intervals = [(df.index[0].to_pydatetime(), df.index[-1].to_pydatetime())]

        return df

    def price_change_query(self):
        start_date = self.start_date
        end_date = self.end_date

        start_date = self.format_date_query(start_date)
        end_date = self.format_date_query(end_date)
        channel_name = f"{self.symbol}_OrderBookFrame"

        query = (
            f"SELECT difference(last(best_ask)) AS change FROM {channel_name} "
            f"WHERE time >= {start_date} AND time <= {end_date} GROUP BY "
            f"time({self.group_by});"
        )

        trades = self.query(query)

        df = pd.DataFrame(columns=["time", "change"])

        for data in trades.get_points(channel_name):
            timestamp = DateTimeUtils.parse_db_timestamp(data["time"])
            data["time"] = timestamp
            df.loc[df.shape[0], :] = data

        return df

    def trade_volume_query(self):
        start_date = self.start_date
        end_date = self.end_date

        start_date = self.format_date_query(start_date)
        end_date = self.format_date_query(end_date)
        channel_name = f"{self.symbol}_trade"

        query = (
            f"SELECT sum(quantity) AS volume FROM {channel_name} "
            f"WHERE time >= {start_date} AND time <= {end_date} GROUP BY "
            f"time({self.group_by});"
        )

        trades = self.query(query)

        df = pd.DataFrame(columns=["time", "volume"])

        for data in trades.get_points(channel_name):
            timestamp = DateTimeUtils.parse_db_timestamp(data["time"])
            data["time"] = timestamp
            df.loc[df.shape[0], :] = data

        return df

    @property
    def price_change_frame(self):
        frames = []

        for interval in self.intervals:
            self.start_date = interval[0]
            self.end_date = interval[1]
            frames.append(self.price_change_query())

        df = pd.concat(frames)

        df.drop_duplicates(subset=["time"], inplace=True)

        df = df.set_index("time")
        df = df.sort_index()
        df.dropna(how="any", inplace=True)

        return df

    @property
    def trade_volume_frame(self):
        frames = []

        for interval in self.intervals:
            self.start_date = interval[0]
            self.end_date = interval[1]
            frames.append(self.trade_volume_query())

        df = pd.concat(frames)

        df.drop_duplicates(subset=["time"], inplace=True)

        df = df.set_index("time")
        df = df.sort_index()
        df.dropna(how="any", inplace=True)

        return df

    @property
    def frame(self):
        if self.cache and self.filename.exists():
            df = pd.read_pickle(str(self.filename))
            self.quantile = df.attrs["quantile"]
            self.trade_volume_max = df.attrs["trade_volume_max"]
            self.change_max = df.attrs["change_max"]
            return df

        df = self._frame()

        orderbook_img = df.orderbook_img.to_numpy().tolist()

        df.drop(["orderbook_img"], axis=1)

        orderbook_img = np.asarray(orderbook_img)

        orderbook_img = np.concatenate(
            (orderbook_img[:, :, 0], orderbook_img[:, :, 1]), axis=2
        )

        orderbook_img = np.absolute(orderbook_img)

        for frame_ix in range(orderbook_img.shape[0]):
            orderbook = orderbook_img[frame_ix]
            shape = orderbook.shape
            new_ob = np.zeros((shape[0], shape[1], 1))

            last_frame_price = orderbook[-1][:, 0]

            for i in range(shape[0]):
                frame = orderbook[i]

                for l in range(frame.shape[0]):
                    level = frame[l]

                    price, volume = level

                    last_frame_index = np.where(last_frame_price == price)

                    new_ob[i, last_frame_index[0], 0] = np.asarray([volume])

            orderbook_img[frame_ix] = new_ob

        orderbook_img = np.delete(orderbook_img, 1, axis=3)

        if self.quantile == 0.0:
            self.quantile = np.quantile(
                orderbook_img.flatten(), self.max_volume_quantile
            )
            if self.quantile == 0.0:
                self.quantile = 1.0

        orderbook_img = orderbook_img / self.quantile

        orderbook_img = np.clip(orderbook_img, a_min=0.0, a_max=1.0)
        # orderbook_img = self.add_price_change(df, orderbook_img)
        # orderbook_img = self.add_trade_volume(df, orderbook_img)

        df["orderbook_img"] = [
            np.rot90(np.fliplr(orderbook_img[i]))
            for i in range(0, orderbook_img.shape[0])
        ]

        df.attrs["trade_volume_max"] = self.trade_volume_max
        df.attrs["change_max"] = self.change_max
        df.attrs["quantile"] = self.quantile

        self.cache_frame(df)

        return df

    def add_price_change(self, df, orderbook_img):
        new_shape = list(orderbook_img.shape)
        new_shape[2] = new_shape[2] + 1
        new_orderbook_img = np.zeros(new_shape)
        old_shape = orderbook_img.shape
        new_orderbook_img[
            : old_shape[0], : old_shape[1], : old_shape[2], : old_shape[3]
        ] = orderbook_img
        orderbook_img = new_orderbook_img

        df = df.join(self.price_change_frame).fillna(0.0)

        # change = self.price_change_frame.reset_index(drop=True)
        change = np.squeeze(df.change.to_numpy())

        if self.change_max == 0.0:
            self.change_max = np.quantile(change, 1.0)

        change = change / self.change_max

        for index in range(len(change)):
            if index <= len(orderbook_img) - 1:
                ord_img = orderbook_img[index]
                frame_size = len(ord_img)
                for i in range(frame_size):
                    change_index = index - i
                    if change_index >= 0:
                        ord_img[i][-1] = [change[change_index]]

                orderbook_img[index] = ord_img

        return orderbook_img

    def add_trade_volume(self, df, orderbook_img):
        height = 6
        df = df.join(self.trade_volume_frame).fillna(0.0)
        trades = df.volume.to_numpy()
        new_shape = list(orderbook_img.shape)
        new_shape[2] = new_shape[2] + height
        new_orderbook_img = np.zeros(new_shape)
        old_shape = orderbook_img.shape
        new_orderbook_img[
            : old_shape[0], : old_shape[1], : old_shape[2], : old_shape[3]
        ] = orderbook_img
        orderbook_img = new_orderbook_img

        if self.trade_volume_max == 0.0:
            self.trade_volume_max = np.quantile(trades, 1.0)

        trades = trades / self.trade_volume_max

        expanded_trades = []

        for vol in trades:
            vol = vol * height
            expanded_vol = np.zeros([height, 1])

            for e in range(expanded_vol.shape[0]):
                if vol > e:
                    expanded_vol[e][0] = 1.0

            expanded_trades.append(expanded_vol)

        trades = np.asarray(expanded_trades)
        trades = np.flip(trades, 1)

        # alog.info(trades)
        # alog.info(orderbook_img.shape)
        # raise Exception()

        for index in range(len(trades)):
            if index <= len(orderbook_img) - 1:
                ord_img = orderbook_img[index]
                frame_size = len(ord_img)
                for i in range(frame_size):
                    trade_index = index - i
                    if trade_index >= 0:
                        # alog.info(ord_img[i])
                        ord_img[i][-height:] = trades[trade_index]

                orderbook_img[index] = ord_img

        return orderbook_img

    @property
    def intervals(self):
        if not self._intervals:
            self.reset_interval()
            offset_interval = timedelta(seconds=timeparse(self.offset_interval))
            start_date = self.start_date - offset_interval
            end_date = self.end_date - offset_interval
            self._intervals = [(start_date, end_date)]

            return self._intervals
        else:
            return self._intervals

    @intervals.setter
    def intervals(self, value):
        self._intervals = value

    def load_frames(self):
        frames = []

        self.start_date = self.start_date - timedelta(
            seconds=(timeparse(self.group_by) * self.sequence_length)
        )

        levels = OrderBookLevelStreamer(
            database_name=self.database_name,
            depth=self.depth,
            end_date=self.end_date,
            group_by=self.group_by,
            start_date=self.start_date,
            symbol=self.symbol,
            window_size=self.window_size,
        )

        df = pd.DataFrame(
            columns=["timestamp", "best_ask", "best_bid", "orderbook_img"]
        )

        for row in levels:
            df.loc[df.shape[0], :] = row

        orderbook_imgs = deque(maxlen=self.sequence_length)

        max_shape = (2, self.output_depth, 2)

        for ix, timestamp, best_ask, best_bid, orderbook_img in df.itertuples():
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
                    left = left[::-1]
                    orderbook_img[0, :left_len, :2] = left[-left_len:, :2]
                    orderbook_img[1, :right_len, :2] = right[:right_len, :2]
                    orderbook_imgs.append(orderbook_img)

                    if len(orderbook_imgs) == self.sequence_length:
                        _orderbook_imgs = np.asarray(list(orderbook_imgs.copy()))

                        _orderbook_imgs[_orderbook_imgs == np.inf] = 0.0
                        _orderbook_imgs[_orderbook_imgs == -np.inf] = 0.0

                        frame = dict(
                            time=timestamp,
                            best_ask=best_ask,
                            best_bid=best_bid,
                            orderbook_img=_orderbook_imgs,
                            dtype=np.float16,
                        )
                        frames.append(frame)

        df = DataFrame(frames)

        df = df.astype({"best_ask": np.float16, "best_bid": np.float16})

        return df

    def cache_frame(self, df):
        df.to_pickle(str(self.filename))

    @staticmethod
    def group_price_levels(orderbook_side):
        groups = dict()

        for price_vol in orderbook_side.tolist():
            price = price_vol[0]
            vol = groups.get(price, 0.0)
            vol += price_vol[1]
            groups[price] = vol

        result = np.array([[p, v] for p, v in groups.items()])

        return result

    def plot_orderbook(self, data):
        fig, frame = plt.subplots(1, 1, figsize=(1, 1), dpi=self.frame_width)
        # frame.axis('off')
        frame = frame.twinx()
        plt.autoscale(tight=True)
        frame.axis("off")
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)

        fig.patch.set_visible(False)
        frame.imshow(data)
        fig.canvas.draw()
        img = fig.canvas.renderer._renderer
        plt.close()
        img = im.fromarray(np.array(img))

        return img


@click.command()
@click.option("--database_name", "-d", default="binance", type=str)
@click.option("--depth", default=72, type=int)
@click.option("--group-by", "-g", default="30s", type=str)
@click.option("--interval", "-i", default="10m", type=str)
@click.option("--offset-interval", "-o", default="3h", type=str)
@click.option("--plot", "-p", is_flag=True)
@click.option("--sequence-length", "-l", default=48, type=int)
@click.option("--round-decimals", "-D", default=4, type=int)
@click.option("--tick", is_flag=True)
@click.option("--cache", is_flag=True)
@click.option("--max-volume-quantile", "-m", default=0.99, type=float)
@click.option("--window-size", "-w", default="3m", type=str)
@click.argument("symbol", type=str)
def main(**kwargs):
    df = OrderBookFrame(**kwargs).frame

    alog.info(df)

    # pd.set_option('display.max_rows', len(df) + 1)

    obook = df.orderbook_img.to_numpy()

    obook = np.squeeze(obook[-1])

    alog.info(obook.tolist())


if __name__ == "__main__":
    main()
