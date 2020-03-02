#! /usr/bin/env python

from bitmex import bitmex
from datetime import timedelta
from exchange_data import settings, Database
from exchange_data.channels import BitmexChannels
from exchange_data.emitters import SignalInterceptor
from exchange_data.emitters.messenger import Messenger
from exchange_data.emitters.prediction_emitter import TradeJob
from exchange_data.trading import Positions
from exchange_data.utils import DateTimeUtils, EventEmitterBase
from time import sleep

import alog
import click
import json


class TradeExecutorUtil(object):
    def parse_position_value(self, value):
        return [position for position in Positions
                if position.value == value][0]


class TradeExecutor(
    TradeJob,
    Messenger,
    Database,
    TradeExecutorUtil,
    SignalInterceptor,
    DateTimeUtils
):

    def __init__(
        self,
        exit_func,
        database_name='bitmex',
        position_size: int = 1,
        **kwargs
    ):
        if exit_func is None:
            exit_func=self.stop

        super().__init__(
            database_name=database_name,
            exit_func=exit_func,
            **kwargs
        )

        self.position_size = position_size
        self.bitmex_client = bitmex(
            test=False,
            api_key=settings.BITMEX_API_KEY,
            api_secret=settings.BITMEX_API_SECRET
        )

        self.on(self.job_name, self.execute)

    @property
    def last_position(self):
        end_date = self.now()
        start_date = end_date - timedelta(hours=24)
        start_date = self.format_date_query(start_date)
        end_date = self.format_date_query(end_date)

        query = f'SELECT LAST(data) as data ' \
            f'FROM position '\
            f'WHERE time > {start_date} AND ' \
            f'time < {end_date} LIMIT 1 tz(\'UTC\');'

        result = self.query(query)
        position = next(result.get_points('position'))

        current_qty = json.loads(position['data'])['data'][0]['currentQty']

        position = None
        if current_qty > 0:
            position = Positions.Long
        else:
            position = Positions.Flat

        return position

    def start(self, channels=[]):
        self.sub([self.job_name] + channels)

    def execute(self, action):
        position = self.parse_position_value(int(action['data']))

        if position.value != self.last_position.value:
            if position == Positions.Flat:
                self.close()
            elif position == Positions.Long:
                self.long()
            elif position == Positions.Short:
                self.short()

        self.last_position = position

    def close(self):
        try:
            self._close()
        except:
            sleep(0.1)
            self.close()

    def _close(self):
        side = None
        if self.last_position == Positions.Long:
            side = 'Sell'
        elif self.last_position == Positions.Short:
            side = 'Buy'
        elif self.last_position == Positions.Flat:
            return

        result = self.bitmex_client.Order.Order_new(
            symbol=self.symbol.value,
            orderQty=self.position_size,
            side=side,
            ordType='Market'
        ).result()[0]

        alog.info(alog.pformat(result))

    def long(self):
        result = self.bitmex_client.Order.Order_new(
            symbol=self.symbol.value,
            ordType='Market',
            orderQty=self.position_size,
            side='Buy'
        ).result()[0]
        alog.info(alog.pformat(result))

    def short(self):
        result = self.bitmex_client.Order.Order_new(
            symbol=self.symbol.value,
            ordType='Market',
            orderQty=self.position_size,
            side='Sell'
        ).result()[0]
        alog.info(alog.pformat(result))

@click.command()
@click.argument(
    'job_name',
    type=str,
    default=None
)
@click.option('--position-size', '-p', type=int, default=1)
@click.argument('symbol', type=click.Choice(BitmexChannels.__members__))
def main(**kwargs):
    time_emitter = TradeExecutor(**kwargs)
    time_emitter.start()


if __name__ == '__main__':
    main()
