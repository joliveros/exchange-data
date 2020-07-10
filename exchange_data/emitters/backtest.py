#!/usr/bin/env python
import random

from exchange_data.data.orderbook_frame import OrderBookFrame
from exchange_data.emitters.prediction_emitter import PredictionBase
from optuna import Trial

import alog
import click
import plotly.graph_objects as go
import tensorflow as tf

from exchange_data.trading import Positions


class BackTest(OrderBookFrame, PredictionBase):
    def __init__(
        self,
        plot=False,
        trial=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.should_plot = plot
        self.capital = 1.0
        self.entry_price = 0.0
        self.trading_fee = (0.075 / 100)
        self.last_position = Positions.Flat
        self.trial: Trial = trial

        self.df = self.frame

        if self.should_plot:
            self.plot()

    def test(self):
        self.capital = 1.0
        df = self.df.copy()
        df.reset_index(drop=False, inplace=True)
        df = df.apply(self.prediction, axis=1)
        df['capital'] = self.capital
        df = df.apply(self.pnl, axis=1)

        # for i in range(0, len(df)):
        #     alog.info(df.loc[i])

        alog.info(df)

        alog.info(self.capital)
        # alog.info(df['capital'].iloc[-1])
        # alog.info(self.capital)

    def load_previous_frames(self, depth):
        pass

    def pnl(self, row):
        exit_price = 0.0
        position = row['position']

        if position == Positions.Long and self.last_position == Positions.Flat:
            self.entry_price = row['best_ask']
            self.last_position = position

        if position == Positions.Flat and self.last_position == Positions.Long:
            exit_price = row['best_bid']

            if self.entry_price > 0.0:
                change = (exit_price - self.entry_price) / self.entry_price
                self.capital = self.capital * (1 + change)
                self.capital = self.capital * (1 - self.trading_fee)
                self.entry_price = 0.0

            self.last_position = position

        row['capital'] = self.capital

        if self.trial:
            self.trial.report(self.capital, row.name)
            tf.summary.scalar('capital', self.capital, step=row.name)

        return row

    # def get_prediction(self):
    #     if random.randint(0, 1) == 0:
    #         return Positions.Flat
    #     else:
    #         return Positions.Long

    def prediction(self, row):
        # alog.info(row)
        self.frames = row['orderbook_img']

        if len(self.frames) == self.sequence_length:
            position = self.get_prediction()
            row['position'] = position

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


@click.command()
@click.option('--database-name', '-d', default='binance', type=str)
@click.option('--depth', '-d', default=40, type=int)
@click.option('--group-by', '-g', default='1m', type=str)
@click.option('--interval', '-i', default='12h', type=str)
@click.option('--plot', '-p', is_flag=True)
@click.option('--sequence-length', '-l', default=48, type=int)
@click.option('--volatility-intervals', '-v', is_flag=True)
@click.option('--volume-max', default=1.0e4, type=float)
@click.option('--window-size', '-w', default='2h', type=str)
@click.argument('symbol', type=str)
def main(**kwargs):
    # start_date = DateTimeUtils.parse_datetime_str('2020-06-30 23:31:00')
    # end_date = DateTimeUtils.parse_datetime_str('2020-07-01 00:53:00')
    #
    # test = BackTest(start_date=start_date, end_date=end_date, **kwargs)

    backtest = BackTest(**kwargs)
    backtest.test()


if __name__ == '__main__':
    main()
