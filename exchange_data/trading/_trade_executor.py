#! /usr/bin/env python

from bitmex import bitmex
from datetime import timedelta
from exchange_data import settings, Database
from exchange_data.channels import BitmexChannels
from exchange_data.emitters import SignalInterceptor
from exchange_data.emitters.messenger import Messenger
from exchange_data.emitters.prediction_emitter import TradeJob
from exchange_data.trading import Positions
from exchange_data.utils import DateTimeUtils
from time import sleep

import alog
import click
import json


class TradeExecutorUtil(object):
    def parse_position_value(self, value):
        return [position for position in Positions
                if position.value == value][0]


class TradeExecutor(
    Messenger,
    Database,
    TradeExecutorUtil,
    SignalInterceptor,
    DateTimeUtils
):

    def __init__(
        self,
        symbol,
        leverage=10,
        exit_func=None,
        database_name='bitmex',
        position_size: int = 1,
        **kwargs
    ):
        if exit_func is None:
            exit_func = self.stop

        super().__init__(
            database_name=database_name,
            exit_func=exit_func,
            **kwargs
        )
        self.leverage = leverage
        self.symbol = symbol
        self.amount = 0.0
        self.best_bid = 0.0
        self.job_name = f'trade_{symbol}'
        self.position = Positions.Flat
        self._position_size = position_size
        self.bitmex_client = bitmex(
            test=False,
            api_key=settings.BITMEX_API_KEY,
            api_secret=settings.BITMEX_API_SECRET
        )

        self.set_leverage()

        self.on(self.job_name, self.execute)
        self.on('tick', self.get_last_position)
        self.on('tick', self.get_wallet)
        self.on('ticker', self.capture_ticker)

    @property
    def position_size(self):
        if self._position_size == 0.0:
            return self.xbt_balance
        else:
            return self._position_size

    @property
    def xbt_balance(self):
        if self.best_bid == 0.0:
            raise Exception()
        return self.amount / (1 / self.best_bid)

    def capture_ticker(self, ticker):
        self.best_bid = ticker['best_bid']

    def set_leverage(self):
        # alog.info(alog.pformat(self.bitmex_client.swagger_spec.resources[
        #                            'Position'].__dict__))
        if self.amount > 0.0:
            result = self.bitmex_client.Position.Position_updateLeverage(
                symbol=self.symbol,
                leverage=self.leverage
            ).result()[0]

            alog.info(alog.pformat(result))

    def get_wallet(self, timestamp):
        end_date = self.now()
        start_date = end_date - timedelta(hours=24)
        start_date = self.format_date_query(start_date)
        end_date = self.format_date_query(end_date)

        query = f'SELECT LAST(data) as data ' \
            f'FROM wallet '\
            f'WHERE time > {start_date} AND ' \
            f'time < {end_date} LIMIT 1 tz(\'UTC\');'

        result = self.query(query)

        wallet = next(result.get_points('wallet'))

        try:
            data = json.loads(wallet['data'])['data'][0]
            self.amount = data['amount']

        except:
            pass

    def get_last_position(self, timestamp):
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

        current_qty = 0.0
        data = json.loads(position['data'])['data']

        if len(data) > 0:
            current_qty = [0]['currentQty']

        position = None

        if current_qty > 0:
            position = Positions.Long
        else:
            position = Positions.Flat

        self.position = position

    def start(self, channels=[]):
        self.sub([self.job_name, 'tick', 'ticker'] + channels)

    def execute(self, action):
        position = self.parse_position_value(int(action['data']))

        if position.value != self.position.value and self.amount > 0.0:
            if position == Positions.Flat:
                self.close()
            elif position == Positions.Long:
                self.long()
            elif position == Positions.Short:
                self.short()

        alog.info({
            'amount': self.amount,
            'position': self.position,
            'leverage': self.leverage,
        })

    def close(self):
        try:
            self._close()
        except:
            sleep(0.1)
            self.close()

    def _close(self):
        side = None
        if self.position == Positions.Long:
            side = 'Sell'
        elif self.position == Positions.Short:
            side = 'Buy'
        elif self.position == Positions.Flat:
            return

        result = self.bitmex_client.Order.Order_new(
            symbol=self.symbol,
            orderQty=self.position_size,
            side=side,
            ordType='Market'
        ).result()[0]

        alog.info(alog.pformat(result))

    def long(self):
        result = self.bitmex_client.Order.Order_new(
            symbol=self.symbol,
            ordType='Market',
            orderQty=self.position_size,
            side='Buy'
        ).result()[0]
        alog.info(alog.pformat(result))

    def short(self):
        result = self.bitmex_client.Order.Order_new(
            symbol=self.symbol,
            ordType='Market',
            orderQty=self.position_size,
            side='Sell'
        ).result()[0]
        alog.info(alog.pformat(result))


@click.command()
@click.option('--position-size', '-p', type=int, default=1)
@click.option('--leverage', '-l', type=int, default=1)
@click.argument('symbol', type=click.Choice(BitmexChannels.__members__))
def main(**kwargs):
    alog.info(alog.pformat(kwargs))

    time_emitter = TradeExecutor(**kwargs)
    time_emitter.start()


if __name__ == '__main__':
    main()
