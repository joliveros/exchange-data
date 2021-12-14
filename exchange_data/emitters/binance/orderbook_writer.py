#!/usr/bin/env python

from exchange_data import Database
from exchange_data.emitters import Messenger, SignalInterceptor
from exchange_data.emitters.binance import BinanceUtils
from exchange_data.utils import DateTimeUtils
import alog
import click
import pickle
import time
import zlib


class OrderBookWriter(
    Messenger,
    BinanceUtils,
    Database,
    SignalInterceptor,
    DateTimeUtils,
):
    def __init__(
        self,
        db_suffix,
        **kwargs
    ):
        futures = kwargs['futures']

        if futures:
            database_name = 'binance_futures'
            stats_prefix = 'futures'
        else:
            database_name = 'binance'

        if len(db_suffix) > 0:
            database_name = database_name + db_suffix

        super().__init__(
            retry_on_timeout=True,
            socket_keepalive=True,
            database_name=database_name,
            database_batch_size=99,
            stats_prefix=stats_prefix,
            **kwargs
        )

        while True:
            obj = self.redis_client.blpop(database_name)
            obj = zlib.decompress(obj[1])
            meas = pickle.loads(obj)
            self.save_measurements(meas)

    def save_measurements(self, measurement, **kwargs):
        try:
            self._save_measurements(measurement, **kwargs)
        except Exception:
            time.sleep(2)
            self.save_measurements(measurement, **kwargs)

    def _save_measurements(self, measurement, **kwargs):
        self.write_points([measurement], time_precision='s', **kwargs)

@click.command()
@click.option('--futures', '-F', is_flag=True)
@click.option('--db-suffix', '-s', type=str, default='')
def main(**kwargs):
    OrderBookWriter(
        **kwargs
    )


if __name__ == '__main__':
    main()
