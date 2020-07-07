#!/usr/bin/env python
from collections import deque
from datetime import timedelta

from dateutil import tz
from optuna import Trial

from exchange_data.emitters.prediction_emitter import PredictionEmitter
from exchange_data.streamers._orderbook_level import OrderBookLevelStreamer
from exchange_data.trading import Positions
from exchange_data.utils import DateTimeUtils
from pandas import DataFrame
from pytimeparse.timeparse import timeparse
import alog
import click
import json
import numpy as np
import pandas
import random
import plotly.graph_objects as go
import tensorflow as tf


class BackTest(PredictionEmitter):
    def __init__(
        self,
        interval,
        window_size,
        symbol,
        plot=False,
        start_date=None,
        end_date=None,
        trial=None,
        group_by='1m',
        **kwargs
    ):
        self.window_size = window_size
        self.group_by = group_by
        self.should_plot = plot
        self.symbol = symbol
        self.trading_enabled = True

        if start_date:
            self.start_date = start_date
        else:
            self.start_date = DateTimeUtils.now() - timedelta(seconds=timeparse(interval))

        if end_date:
            self.end_date = end_date
        else:
            self.end_date = DateTimeUtils.now()

        self.capital = 1.0
        self.entry_price = 0.0
        self.trading_fee = (0.075 / 100)

        super().__init__(symbol=symbol, **kwargs)

        self.frames = deque(maxlen=self.sequence_length)
        self.group_by_min = int(timeparse(group_by) * 3 / 60)
        self.last_position = None
        self.trial: Trial = trial

        df = self.load_frames(self.depth)
        df.dropna(how='any', inplace=True)
        alog.info(df)
        df['time'] = pandas.to_datetime(df['time'], unit='ms')
        df = df.set_index('time')
        df = df.sort_index()
        df.reset_index(drop=False, inplace=True)

        self.df = df.copy()

        if self.should_plot:
            self.plot()

    def test(self):
        self.capital = 1.0
        df = self.df.copy()
        df = df.apply(self.prediction, axis=1)
        df['capital'] = self.capital
        df = df.apply(self.pnl, axis=1)

        alog.info(df)
        # alog.info(df['capital'].iloc[-1])
        # alog.info(self.capital)

    def load_previous_frames(self, depth):
        pass

    def pnl(self, row):
        exit_price = 0.0
        capital = self.capital
        position = row['position']

        if position > 0 and self.entry_price == 0.0:
            self.entry_price = row['best_ask']

        if position != self.last_position and self.entry_price > 0.0 and \
            self.last_position >= 0.0:
            exit_price = row['best_bid']
            change = (exit_price - self.entry_price) / self.entry_price
            fee = (1 - self.trading_fee)
            capital = capital * fee
            capital = (change + 1) * capital

        self.capital = capital

        row['capital'] = capital

        if self.trial:
            self.trial.report(capital, row.name)
            tf.summary.scalar('capital', capital, step=row.name)

        self.entry_price = 0.0

        self.last_position = position

        return row

    # def get_prediction(self):
    #     if random.randint(0, 1) == 0:
    #         return Positions.Flat
    #     else:
    #         return Positions.Long

    def prediction(self, row):
        # alog.info(row)
        frame = row['orderbook_img']

        self.frames.append(frame)

        if len(self.frames) == self.sequence_length:
            position = self.get_prediction()
            alog.info(position)
            row['position'] = int(position.value)

        return row

    def plot(self):
        df = self.ohlc
        df.reset_index(drop=False, inplace=True)

        alog.info(df)

        fig = go.Figure()

        fig.update_layout(
            yaxis4=dict(
                anchor="free",
                overlaying="y",
                side="left",
                position=0.001
            ),
            yaxis2=dict(
                anchor="free",
                overlaying="y",
                side="right",
                position=0.001
            ),
            yaxis3=dict(
                anchor="free",
                overlaying="y",
                side="right",
                position=0.001
            ),
        )
        fig.add_trace(go.Candlestick(x=df['time'],
                                     open=df['open'],
                                     high=df['high'],
                                     low=df['low'],
                                     close=df['close'], yaxis='y4'))
        # fig.add_trace(go.Scatter(x=df['time'], y=macd, yaxis='y3'))
        # fig.add_trace(go.Scatter(x=df['time'], y=exp3, yaxis='y3'))
        # fig.add_trace(go.Scatter(x=df['time'], y=df['volatility'], yaxis='y2'))
        # fig.add_trace(go.Scatter(
        #     x=minDf['time'],
        #     y=minDf['avg'],
        #     mode='markers',
        #     marker=dict(color="crimson", size=12)
        # ))
        # fig.add_trace(go.Scatter(
        #     x=maxDf['time'],
        #     y=maxDf['avg'],
        #     mode='markers',
        #     marker=dict(color="blue", size=12)
        # ))
        # fig.add_trace(go.Scatter(
        #     x=maxDf['time'],
        #     y=maxDf['avg'],
        #     mode='markers',
        #     marker=dict(color="blue", size=12)
        # ))
        fig.show()

    @property
    def ohlc(self):
        df = self.df
        df['openbid'] = (df['best_ask'] + df['best_bid']) / 2
        ohlc_df = df.drop(df.columns.difference(['time', 'openbid']), 1,
                          inplace=False)
        ohlc_df = ohlc_df.set_index('time')
        alog.info(ohlc_df)
        ohlc_df = ohlc_df.resample(f'{self.group_by_min}T').ohlc()
        ohlc_df.columns = ohlc_df.columns.droplevel()
        ohlc_df = ohlc_df[ohlc_df.low != 0.0]

        return ohlc_df

    def load_frames(self, depth):
        frames = []

        levels = OrderBookLevelStreamer(
            database_name=self.database_name,
            depth=depth,
            end_date=self.end_date,
            groupby=self.group_by,
            start_date=self.start_date,
            symbol=self.symbol,
            window_size=self.window_size
        )

        for timestamp, best_ask, best_bid, orderbook_img in levels:
            if orderbook_img:
                orderbook_img = np.asarray(json.loads(orderbook_img))
                orderbook_img = self.normalize_frame(orderbook_img)
                frame = dict(
                    time=timestamp,
                    best_ask=best_ask,
                    best_bid=best_bid,
                    orderbook_img=orderbook_img
                )
                frames.append(frame)
                # try:
                #     orderbook_img = self.normalize_frame(orderbook_img)
                #     frame = dict(
                #         time=timestamp,
                #         best_ask=best_ask,
                #         best_bid=best_bid,
                #         orderbook_img=orderbook_img
                #     )
                #     frames.append(frame)
                # except:

        return DataFrame(frames)

@click.command()
@click.option('--depth', '-d', default=40, type=int)
@click.option('--sequence-length', '-l', default=48, type=int)
@click.option('--database-name', '-d', default='binance', type=str)
@click.option('--interval', '-i', default='1d', type=str)
@click.option('--window-size', '-w', default='2h', type=str)
@click.option('--group-by', '-g', default='1m', type=str)
@click.option('--volume-max', '-v', default=1.0e4, type=float)
@click.option('--plot', '-p', is_flag=True)
@click.argument('symbol', type=str)
def main(**kwargs):
    # start_date = DateTimeUtils.parse_datetime_str('2020-06-30 23:31:00')
    # end_date = DateTimeUtils.parse_datetime_str('2020-07-01 00:53:00')
    #
    # test = BackTest(start_date=start_date, end_date=end_date, **kwargs)

    BackTest(**kwargs)

if __name__ == '__main__':
    main()
