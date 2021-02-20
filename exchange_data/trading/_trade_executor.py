#! /usr/bin/env python

from bitmex import bitmex
from datetime import timedelta
from exchange_data import settings, Database
from exchange_data.channels import BitmexChannels
from exchange_data.emitters import SignalInterceptor
from exchange_data.emitters.messenger import Messenger
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
        self.balance = 0.0
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
        self.on('2s', self.get_wallet_amount)
        self.on('2s', self.get_position)
        self.on('ticker', self.capture_ticker)
        # self.balance = 1000000
        # alog.info((self.balance, self.leverage))
        # alog.info(alog.pformat(self.bitmex_client.swagger_spec.resources[
        #                            'Position'].__dict__))
        # raise Exception()
        # self.position = Positions.Short

    @property
    def position_size(self):
        if self._position_size == 0.0:
            return self.xbt_balance
        else:
            return self._position_size

    @property
    def xbt_balance(self):
        if self.leverage > 1.0:
            return int((self.balance * self.leverage * 0.9066) / 10000)
        else:
            return int(self.balance / 10000 * 0.9)

    def capture_ticker(self, ticker):
        self.best_bid = ticker['best_bid']

    def get_position(self, timestamp):
        result = self.bitmex_client.Position.Position_get().result()[0][0]
        currentQty = result['currentQty']

        if currentQty < 0:
            self.position = Positions.Short
        else:
            self.position = Positions.Flat

    def set_leverage(self):
        # alog.info(alog.pformat(self.bitmex_client.swagger_spec.resources[
        #                            'Position'].__dict__))

        if self.balance > 0.0:
            result = self.bitmex_client.Position.Position_updateLeverage(
                symbol=self.symbol,
                leverage=self.leverage
            ).result()[0]


            alog.info(alog.pformat(result))

    def get_wallet_amount(self, timestamp):
        result = self.bitmex_client.User.User_getWalletSummary().result()[0]
        last_wallet = result.pop()
        self.balance = last_wallet['walletBalance']

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
            self.balance = data['amount']

        except:
            pass

    # def get_last_position(self, timestamp):
    #     end_date = self.now()
    #     start_date = end_date - timedelta(hours=24)
    #     start_date = self.format_date_query(start_date)
    #     end_date = self.format_date_query(end_date)
    #
    #     query = f'SELECT LAST(data) as data ' \
    #         f'FROM position '\
    #         f'WHERE time > {start_date} AND ' \
    #         f'time < {end_date} LIMIT 1 tz(\'UTC\');'
    #
    #     result = self.query(query)
    #     position = next(result.get_points('position'))
    #
    #     current_qty = 0.0
    #     data = json.loads(position['data'])['data']
    #
    #     if len(data) > 0:
    #         current_qty = data[0]['currentQty']
    #
    #     position = None
    #
    #     if current_qty > 0:
    #         position = Positions.Long
    #     else:
    #         position = Positions.Flat
    #
    #     self.position = position

    def start(self, channels=[]):
        self.sub([self.job_name, '2s', 'ticker'] + channels)

    def execute(self, action):
        position = self.parse_position_value(int(action['data']))

        if self.xbt_balance == 0:
            return

        if position != self.position and self.balance > 0.0:
            if position == Positions.Flat:
                self.close()
            elif position == Positions.Long:
                self.long()
            elif position == Positions.Short:
                self.short()

        alog.info({
            'amount': self.balance,
            'xbt_balance': self.xbt_balance,
            'position': self.position,
            'leverage': self.leverage,
        })

    def close(self):
        try:
            self._close()
        except:
            sleep(0.1)
            self._close()

    def _close(self):
        alog.info(self.position)
        alog.info(self.position == Positions.Flat)

        if self.position == Positions.Flat:
           return

        side = None

        alog.info(f'#### close trade ####')

        if self.position == Positions.Long:
            side = 'Buy'
        elif self.position == Positions.Short:
            side = 'Sell'

        result = self.bitmex_client.Order.Order_closePosition(
            symbol=self.symbol,
        ).result()[0]
        # result = self.bitmex_client.Order.Order_new(
        #     execInst='Close',
        #     symbol=self.symbol,
        #     # side=side,
        #     # ordType='Market'
        # ).result()[0]

        alog.info(alog.pformat(result))

        self.position = Positions.Flat

    def long(self):
        # result = self.bitmex_client.Order.Order_new(
        #     symbol=self.symbol,
        #     ordType='Market',
        #     orderQty=self.position_size,
        #     side='Buy'
        # ).result()[0]
        # alog.info(alog.pformat(result))
        self.position = Positions.Long

    def short(self):
        alog.info(self.position)

        if self.position != Positions.Short:
            self.close()

            alog.info('### initiate short ####')
            alog.info(self.position_size)

            result = self.bitmex_client.Order.Order_new(
                symbol=self.symbol,
                ordType='Market',
                orderQty=self.position_size,
                side='Sell'
            ).result()[0]

            alog.info(alog.pformat(result))

        self.position = Positions.Short


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
