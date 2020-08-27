#!/usr/bin/env python

from binance.client import Client
from cached_property import cached_property, cached_property_with_ttl
from exchange_data import settings, Database
from exchange_data.data.measurement_frame import MeasurementFrame
from exchange_data.emitters import Messenger
from exchange_data.ta_model.tune_macd import MacdParamsFrame

import alog
import click
import signal
import sys


class AssetPair(object):
    def __init__(self, asset, base_asset, **kwargs):
        # super().__init__(
        #     **kwargs
        # )
        self.base_asset = base_asset
        self.asset = asset

        # params = MacdParams(database_name=self.database_name)

    @property
    def symbol(self):
        return self.asset + self.base_asset


class Position(Database, AssetPair):
    measurements = []

    def __init__(self, **kwargs):
        super().__init__(
            **kwargs
        )

        params_df = MacdParamsFrame(**kwargs).frame_all_keys()

        params_df.dropna(how='any', inplace=True)

        alog.info(params_df)


@click.command()
@click.option('--database-name', '-d', default='binance', type=str)
@click.option('--base-asset', '-b', default='BNB', type=str)
@click.option('--asset', '-a', default='ALGO', type=str)
@click.option('--interval', '-i', default='2h', type=str)
def main(**kwargs):
    emitter = Position(**kwargs)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda: exit(0))
    signal.signal(signal.SIGTERM, lambda: exit(0))
    main()
