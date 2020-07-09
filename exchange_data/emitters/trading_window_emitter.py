#!/usr/bin/env python
import json
from collections import OrderedDict
from datetime import timedelta

import pytz

from exchange_data import Database
from exchange_data.channels import BitmexChannels
from exchange_data.emitters import Messenger
from exchange_data.utils import DateTimeUtils
from pandas import DataFrame
from pytimeparse.timeparse import timeparse

import alog
import click
import matplotlib.pyplot as plt
import pandas
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots


class TradingWindowEmitter(Messenger, Database, DateTimeUtils):
    def __init__(
        self,
        symbol,
        interval='2h',
        group_by='1m',
        plot=False,
        **kwargs
    ):
        super().__init__(
            database_batch_size=1,
            **kwargs
        )

        self.should_plot = plot
        self.symbol = symbol
        self.channels += ['2s']
        self.group_by_min = int(timeparse(group_by)/60)
        self.group_by = f'{int(timeparse(group_by)/3)}s'
        alog.info(interval)
        self.interval_delta = timedelta(seconds=timeparse(interval))

        self.channel_name = 'should_trade'
        self.frame_channel = f'{symbol}_OrderBookFrame_depth_40'

        self.on('2s', self.next_intervals)

    def next_intervals(self, timestamp=None):
        rows = [(tick['time'], tick['data']) for tick in self.query_price()[
            self.frame_channel]]
        df = DataFrame(rows, columns=['time', 'openbid'])
        df.dropna(how='any', inplace=True)
        df['time'] = pandas.to_datetime(df['time'], unit='ms')
        df = df.set_index('time')
        df = df.resample(f'{self.group_by_min}T').ohlc()
        df = df.reset_index(drop=False, col_level=1)
        df.columns = df.columns.droplevel()
        df = df[df.low != 0.0]

        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df.dropna(how='any', inplace=True)

        window = 31
        df['volatility'] = df['log_ret'].rolling(window=window).std() * np.sqrt(
            window)

        dfv = df.copy()
        dfv = dfv.drop(columns=['open', 'high', 'low', 'close', 'log_ret'])

        dfv = dfv[dfv.volatility > 0.0015]

        required_volatility_ts = tuple(dfv.time)

        df_close = df['close']
        exp1 = df_close.ewm(span=12, adjust=False).mean()
        exp2 = df_close.ewm(span=26, adjust=False).mean()
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

        minmax_pairs = np.asarray(sorted(
            tuple(zip(maxDf.time, maxDf.avg, maxDf.type)) + tuple(
                zip(minDf.time, minDf.avg, minDf.type))))

        intervals = []
        last_index = minmax_pairs.shape[0] - 1

        for i in range(minmax_pairs.shape[0]):
            t = minmax_pairs[i]

            if t[-1] == 'min':
                if i != last_index:
                    interval = (t[0], minmax_pairs[i + 1][0])
                    intervals += [interval]

            if i == last_index:
                if t[-1] == 'min':
                    alog.info('## should trade ##')
                    self.publish(self.channel_name, json.dumps({'should_trade': str(True)}))
                else:
                    alog.info('## should not trade ##')
                    self.publish(self.channel_name, json.dumps({'should_trade': str(
                        False)}))

        self.df = df
        self.exp3 = exp3
        self.macd = macd
        self.maxDf = maxDf
        self.minDf = minDf

        # intervals = [i for i in intervals if i[0] in required_volatility_ts]

        intervals = [(pytz.utc.localize(i[0].to_pydatetime()),
                      pytz.utc.localize(i[1].to_pydatetime())) for i in
                     intervals]

        self.intervals = intervals

        if self.should_plot:
            self.plot()

    def plot(self):
        df = self.df
        alog.info(df)
        exp3 = self.exp3
        macd = self.macd
        maxDf = self.maxDf
        minDf = self.minDf

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
        fig.add_trace(go.Scatter(x=df['time'], y=macd, yaxis='y3'))
        fig.add_trace(go.Scatter(x=df['time'], y=exp3, yaxis='y3'))
        fig.add_trace(go.Scatter(x=df['time'], y=df['volatility'], yaxis='y2'))
        fig.add_trace(go.Scatter(
            x=minDf['time'],
            y=minDf['avg'],
            mode='markers',
            marker=dict(color="crimson", size=12)
        ))
        fig.add_trace(go.Scatter(
            x=maxDf['time'],
            y=maxDf['avg'],
            mode='markers',
            marker=dict(color="blue", size=12)
        ))
        fig.add_trace(go.Scatter(
            x=maxDf['time'],
            y=maxDf['avg'],
            mode='markers',
            marker=dict(color="blue", size=12)
        ))
        fig.show()

    def query_price(self):
        now = DateTimeUtils.now()
        start_date = now - self.interval_delta
        end_date = now

        start_date = self.format_date_query(start_date)
        end_date = self.format_date_query(end_date)

        query = f'SELECT first(best_ask) AS data FROM {self.frame_channel} ' \
            f'WHERE time > {start_date} AND time <= {end_date} GROUP BY time(' \
                f'{self.group_by});'

        alog.info(query)

        return self.query(query)

    def run(self):
        self.sub(self.channels)


@click.command()
@click.option('--interval', '-i', default='1h', type=str)
@click.option('--database-name', '-d', default='binance', type=str)
@click.option('--symbol', '-s', default='', type=str)
@click.option('--group-by', '-g', default='1m', type=str)
@click.option('--once', '-o', is_flag=True)
@click.option('--plot', '-p', is_flag=True)
def main(once, **kwargs):
    record = TradingWindowEmitter(**kwargs)
    if once:
        record.next_intervals()
    else:
        record.run()


if __name__ == '__main__':
    main()
