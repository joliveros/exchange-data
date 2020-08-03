#!/usr/bin/env python

from cached_property import cached_property
from pytimeparse.timeparse import timeparse
from redis_collections import Set
from redlock import RedLock, RedLockError

from exchange_data import Database, Measurement, settings
from exchange_data.data.price_frame import PriceFrame
from exchange_data.emitters import Messenger
from exchange_data.emitters.backtest_base import BackTestBase
from exchange_data.ta_model.backtest import TuneMACDSignal
from exchange_data.trading import Positions
from optuna import create_study, Trial
from pandas import DataFrame

import alog
import click
import json
import pandas as pd



class TuneSymbols(Database, Messenger):
    def __init__(self, base_currency, **kwargs):
        super().__init__(**kwargs)
        self.base_currency = base_currency
        self.symbols_queue = Set(key='symbol_tune_queue',
                                 redis=self.redis_client)

        alog.info(self.symbols_queue)

        self.reset_queue()

        while len(self.symbols_queue) > 0:
            symbol = self.symbols_queue.pop()
            TuneMACDSignal(symbol=symbol, **kwargs)


    def reset_queue_lock(self):
        lock_name = 'reset_queue_lock'

        return RedLock(lock_name, [dict(
            host=settings.REDIS_HOST,
            db=0
        )], retry_delay=200, retry_times=3, ttl=timeparse('1m') * 1000)

    @cached_property
    def symbols(self):
        result = self.query(f'SHOW MEASUREMENTS ON {self.database_name}')

        measurements = [d['name'] for d in result['measurements']]

        symbols = [m.split('_')[0] for m in measurements
                   if m.endswith('OrderBookFrame')]

        symbols = [s for s in symbols if s.endswith(self.base_currency)]

        return symbols

    def reset_queue(self):
        try:
            self._reset_queue()
        except RedLockError as e:
            alog.info(e)

    def _reset_queue(self):
        with self.reset_queue_lock():
            for s in self.symbols:
                self.symbols_queue.add(s)



@click.command()
@click.option('--database-name', '-d', default='binance', type=str)
@click.option('--group-by', '-g', default='1m', type=str)
@click.option('--interval', '-i', default='12h', type=str)
@click.option('--window-size', '-w', default='2h', type=str)
@click.option('--base-currency', '-c', default='BNB', type=str)
@click.option('--session-limit', '-s', default=100, type=int)
def main(**kwargs):
    TuneSymbols(**kwargs)


if __name__ == '__main__':
    main()
