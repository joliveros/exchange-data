from exchange_data.trading import Positions
from plotly import graph_objects as go
import tensorflow as tf
import alog


class BackTestBase(object):
    capital = 1.0

    def __init__(
        self,
        group_by_min=1,
        print_all_rows=False,
        plot=False,
        **kwargs
    ):
        self.print_all_rows = print_all_rows
        self.trial = None
        self.group_by_min = group_by_min

        self.should_plot = plot
        self.entry_price = 0.0
        self.trading_fee = (0.075 / 100)
        self.last_position = Positions.Flat

        if self.should_plot:
            self.plot()

    @property
    def frame(self):
        raise NotImplemented()

    def test(self):
        raise NotImplemented()

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
                self.capital = self.capital * ((1 - self.trading_fee) ** 2)
                self.entry_price = 0.0

            self.last_position = position

        row['capital'] = self.capital

        if self.trial:
            # self.trial.report(self.capital, row.name)
            tf.summary.scalar('capital', self.capital, step=row.name)

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
        df = self.frame.copy()
        df.reset_index(drop=False, inplace=True)
        df['openbid'] = (df['best_ask'] + df['best_bid']) / 2
        ohlc_df = df.drop(df.columns.difference(['time', 'openbid']), 1,
                          inplace=False)
        ohlc_df = ohlc_df.set_index('time')
        ohlc_df = ohlc_df.resample(f'{self.group_by_min}T').ohlc()
        ohlc_df.columns = ohlc_df.columns.droplevel()
        ohlc_df = ohlc_df[ohlc_df.low != 0.0]

        # alog.info(ohlc_df)

        return ohlc_df
