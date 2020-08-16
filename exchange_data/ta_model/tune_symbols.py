#!/usr/bin/env python

from cached_property import cached_property
from exchange_data import Database, settings
from exchange_data.emitters import Messenger
from exchange_data.emitters.binance.volatility_change_emitter import VolatilityChangeEmitter
from exchange_data.ta_model.backtest import TuneMACDSignal
from pytimeparse.timeparse import timeparse
from redis_collections import Set
from redlock import RedLock, RedLockError

import alog
import click


class TuneSymbols(Database, Messenger):
    def __init__(self, base_currency, symbol_limit, clear=False, **kwargs):
        super().__init__(**kwargs)
        self.symbol_limit = symbol_limit
        self.base_currency = base_currency
        self.symbols_queue = Set(key='symbol_tune_queue',
                                 redis=self.redis_client)

        if clear:
            self.symbols_queue.clear()

        alog.info(self.symbols_queue)

        self.reset_queue()

        while len(self.symbols_queue) > 0:
            symbol = self.symbols_queue.pop()

            try:
                TuneMACDSignal(symbol=symbol, **kwargs)
            except KeyError:
                pass

        alog.info('## should exit ##')

    def reset_queue_lock(self):
        lock_name = 'reset_queue_lock'

        return RedLock(lock_name, [dict(
            host=settings.REDIS_HOST,
            db=0
        )], retry_delay=200, retry_times=3, ttl=timeparse('1m') * 1000)

    @cached_property
    def symbols(self):
        volatility_change = VolatilityChangeEmitter(
            database_name='binance',
            filter='BNB',
            interval='30m', group_by='15m'
        )
        return volatility_change.frame()[-1].head(
            self.symbol_limit).pair.to_list()

    def reset_queue(self):
        try:
            self._reset_queue()
        except RedLockError as e:
            alog.info(e)

    def _reset_queue(self):
        with self.reset_queue_lock():
            for s in self.symbols:
                alog.info(s)
                self.symbols_queue.add(s)



@click.command()
@click.option('--base-currency', '-c', default='BNB', type=str)
@click.option('--clear', is_flag=True)
@click.option('--database-name', '-d', default='binance', type=str)
@click.option('--group-by', '-g', default='1m', type=str)
@click.option('--interval', '-i', default='12h', type=str)
@click.option('--session-limit', '-s', default=100, type=int)
@click.option('--symbol-limit', default=15, type=int)
@click.option('--window-size', '-w', default='2h', type=str)
def main(**kwargs):
    TuneSymbols(**kwargs)


if __name__ == '__main__':
    main()
