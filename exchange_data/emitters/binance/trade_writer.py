#!/usr/bin/env python
import json

from exchange_data import Database, Measurement
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
        batch_size,
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
            database_batch_size=batch_size,
            stats_prefix=stats_prefix,
            **kwargs
        )
        list_name = f'{database_name}_trades'

        while True:
            batch_size = int(self.redis_client.llen(list_name) * 0.99)
            if batch_size > 0:
                self.batch_size = batch_size
            else:
                self.batch_size = 1000

            obj = self.redis_client.blpop(list_name)
            obj = zlib.decompress(obj[1])
            meas = pickle.loads(obj)
            self.save_measurements(meas)

    def save_measurements(self, measurement, **kwargs):
        try:
            self._save_measurements(measurement, **kwargs)
        except Exception as e:
            alog.info(e)
            time.sleep(2)
            self.save_measurements(measurement, **kwargs)

    def _save_measurements(self, measurement, **kwargs):
        timestamp = DateTimeUtils.parse_db_timestamp(measurement['T'])

        measurement = Measurement(fields={
            'price': float(measurement['p']),
            'quantity': float(measurement['q']),
            'market-maker': measurement['m']
            },
            measurement=f'{measurement["s"]}_trade',
            time=timestamp
        )

        self.write_points([measurement.__dict__], time_precision='s', consistency='all', **kwargs)

@click.command()
@click.option('--futures', '-F', is_flag=True)
@click.option('--db-suffix', '-s', type=str, default='')
@click.option('--batch-size', '-b', type=int, default=1000)
def main(**kwargs):
    OrderBookWriter(
        **kwargs
    )


if __name__ == '__main__':
    main()
