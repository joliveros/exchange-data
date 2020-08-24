#!/usr/bin/env python

from exchange_data.data.measurement_frame import MeasurementFrame
from exchange_data.ta_model.backtest import BackTest
from optuna import create_study, Trial

import alog
import click


class MacdParamFrame(MeasurementFrame):
    pass


class TuneMACDSignal(object):
    study = None

    def __init__(self, symbol, database_name, n_jobs=8, session_limit=100,
                 **kwargs):
        kwargs['symbol'] = symbol
        kwargs['database_name'] = database_name

        self._kwargs = kwargs
        self.n_jobs = n_jobs
        self.database_name = database_name
        self.symbol = symbol
        self.session_limit = session_limit

    def run_study(self):
        self.study = create_study(
            study_name=self.symbol, direction='maximize')

        self.study.optimize(self.run, n_trials=self.session_limit,
                            n_jobs=self.n_jobs)

        alog.info(alog.pformat(vars(self.study.best_trial)))

        data = dict(
            **self.study.best_trial.params,
            symbol=self.symbol,
            value=self.study.best_trial.value,
        )

        MacdParamFrame(database_name=self.database_name).append(data)

    def run(self, trial: Trial):
        params = dict(
            short_period=trial.suggest_int('short_period', 1, 96 * 3),
            long_period=trial.suggest_int('long_period', 1, 96 * 3),
        )

        return BackTest(**self._kwargs).test(**params)


@click.command()
@click.option('--database-name', '-d', default='binance', type=str)
@click.option('--group-by', '-g', default='1m', type=str)
@click.option('--group-by-min', '-m', default='4', type=str)
@click.option('--interval', '-i', default='12h', type=str)
@click.option('--plot', '-p', is_flag=True)
@click.option('--window-size', '-w', default='2h', type=str)
@click.option('--session-limit', '-s', default=100, type=int)
@click.option('--n-jobs', '-n', default=8, type=int)
@click.argument('symbol', type=str)
def main(**kwargs):
    TuneMACDSignal(**kwargs)


if __name__ == '__main__':
    main()
