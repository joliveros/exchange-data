#!/usr/bin/env python
from exchange_data.data.labeled_orderbook_frame import LabeledOrderBookFrame
from exchange_data.data.orderbook_frame import OrderBookFrame
from exchange_data.emitters.backtest_base import BackTestBase
from exchange_data.emitters.prediction_emitter import PredictionBase
from optuna import Trial

import alog
import click


class BackTest(OrderBookFrame, BackTestBase, PredictionBase):
    def __init__(
        self,
        trial=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        BackTestBase.__init__(self, **kwargs)
        PredictionBase.__init__(self, **kwargs)

        self.trial: Trial = trial

    def test(self, model_version=None):
        if model_version:
            self.model_version = model_version

        self.capital = 1.0
        df = self.frame.copy()
        df.reset_index(drop=False, inplace=True)
        df = df.apply(self.prediction, axis=1)
        df['capital'] = self.capital

        df = df.apply(self.pnl, axis=1)

        return df

    def load_previous_frames(self, depth):
        pass

    # def get_prediction(self):
    #     if random.randint(0, 1) == 0:
    #         return Positions.Flat
    #     else:
    #         return Positions.Long

    def prediction(self, row):
        self.frames = row['orderbook_img']

        if len(self.frames) == self.sequence_length:
            position = self.get_prediction()
            row['position'] = position

        return row



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
@click.option('--model-version', default=None, type=str)
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
