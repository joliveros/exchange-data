#!/usr/bin/env python

from exchange_data.ta_model.single_pass_backtest import SinglePassBackTest
from exchange_data.ta_model.tune_macd import TuneMACDSignal
from optuna import Trial

import alog
import click


class TuneMACDSignalSinglePass(TuneMACDSignal):
    def __init__(self, **kwargs):
        self.backtest = SinglePassBackTest(**kwargs)

        super().__init__(**kwargs)

    def run(self, trial: Trial):

        params = dict(
            span_1=trial.suggest_int('span_1', 1, 2 ** 8),
            span_2=trial.suggest_int('span_2', 1, 2 ** 8),
            span_3=trial.suggest_int('span_3', 1, 2 ** 8),
        )

        return self.backtest.test(**params)



@click.command()
@click.option('--database-name', '-d', default='binance', type=str)
@click.option('--group-by', '-g', default='1m', type=str)
@click.option('--group-by-min', '-m', default='4', type=str)
@click.option('--interval', '-i', default='12h', type=str)
@click.option('--plot', '-p', is_flag=True)
@click.option('--window-size', '-w', default='2h', type=str)
@click.option('--session-limit', '-s', default=100, type=int)
@click.option('--n-jobs', '-n', default=1, type=int)
@click.argument('symbol', type=str)
def main(**kwargs):
    result = TuneMACDSignalSinglePass(**kwargs).run_study()

    alog.info(alog.pformat(result))


if __name__ == '__main__':
    main()
