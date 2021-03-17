#!/usr/bin/env python

from exchange_data.data.orderbook_frame import OrderBookFrame
from exchange_data.emitters.backtest_base import BackTestBase
from exchange_data.trading import Positions
from plotly import graph_objects as go

import alog
import click
import pandas as pd
import numpy as np

pd.options.plotting.backend = 'plotly'


class MaxMinFrame(OrderBookFrame, BackTestBase):
    def __init__(
        self,
        symbol,
        negative_change_quantile=0.7,
        positive_change_quantile=0.7,
        flat_ratio=0.5,
        **kwargs
    ):
        self.positive_change_quantile = positive_change_quantile
        self.negative_change_quantile = negative_change_quantile
        self.flat_ratio = flat_ratio

        super().__init__(symbol=symbol, **kwargs)
        BackTestBase.__init__(self, symbol=symbol, **kwargs)

    def label_position(self):
        df = self.ohlc.copy()

        df.reset_index(drop=False, inplace=True)

        df['position'] = Positions.Flat

        df['change'] = df.close - df.open

        #pd.set_option('display.max_rows', len(df) + 1)

        df['change'] = df['change'].fillna(0.0)

        change = df['change'].to_numpy()

        pos_change = np.where(change < 0.0, 0, change)
        neg_change = np.abs(np.where(change > 0.0, 0, change))
        pos_change_quantile = np.quantile(pos_change,
                                          self.positive_change_quantile)
        neg_change_quantile = np.quantile(neg_change,
                                          self.negative_change_quantile)

        pos_change = np.where(pos_change > pos_change_quantile,
                              Positions.Long, pos_change)

        neg_change = np.where(neg_change > neg_change_quantile,
                              Positions.Flat, neg_change)

        position = np.zeros(pos_change.shape, dtype=Positions)

        for i in range(0, position.shape[0]):
            pos = pos_change[i]

            if pos != 0.0:
                position[i] = pos

            pos = neg_change[i]

            if pos != 0.0:
                position[i] = pos

        for i in range(0, position.shape[0]):
            if type(position[i]) != Positions:
                position[i] = 0

        df = df.drop(['change'], axis=1)

        df['position'] = position

        flat_df = df[df['position'] == 0]\
            .sample(frac=self.flat_ratio, random_state=0)

        flat_df['position'] = Positions.Flat

        df = df[df['position'] != 0]
        df = pd.concat([df, flat_df])

        df = df.set_index('time')

        return df

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

        df = self.label_position().copy()

        df['position'] = df['position'].replace([Positions.Long], 1)
        df['position'] = df['position'].replace([Positions.Flat], 0)

        alog.info(df)

        fig.add_trace(go.Scatter(x=df.index, y=df['position'], mode='lines'))

        fig.show()

    def label_positive_change(
        self,
        **kwargs
    ):
        position = self.label_position().position

        df = self.frame.copy()

        position = position.replace(Positions.Long, 1)
        position = position.replace(Positions.Flat, 0)

        df['expected_position'] = position

        df.dropna(how='any', inplace=True)

        alog.info(f'### volume_max {self.quantile} ####')

        return df


@click.command()
@click.option('--database_name', '-d', default='binance', type=str)
@click.option('--depth', default=40, type=int)
@click.option('--group-by', '-g', default='1m', type=str)
@click.option('--group-by-min', '-G', default='1m', type=str)
@click.option('--interval', '-i', default='3h', type=str)
@click.option('--max-volume-quantile', '-m', default=0.99, type=float)
@click.option('--negative-change-quantile', '-n', default=0.50, type=float)
@click.option('--offset-interval', '-o', default='3h', type=str)
@click.option('--plot', '-p', is_flag=True)
@click.option('--positive-change-quantile', '-q', default=0.50, type=float)
@click.option('--round-decimals', '-D', default=3, type=int)
@click.option('--sequence-length', '-l', default=48, type=int)
@click.option('--tick', is_flag=True)
@click.option('--volatility-intervals', '-v', is_flag=True)
@click.option('--window-size', '-w', default='3m', type=str)
@click.argument('symbol', type=str)
def main(**kwargs):
    # df = MaxMinFrame(**kwargs)
    # return

    df = MaxMinFrame(**kwargs).label_positive_change()

    pd.set_option('display.max_rows', len(df) + 1)

    alog.info(df)



if __name__ == '__main__':
    main()
