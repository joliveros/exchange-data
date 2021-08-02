#!/usr/bin/env python
import math
from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET
from cached_property import cached_property_with_ttl
from decimal import Decimal, getcontext
from exchange_data import settings
from exchange_data.emitters import binance, Messenger
from exchange_data.emitters.binance import BinanceUtils
from exchange_data.models.resnet.study_wrapper import StudyWrapper
from exchange_data.trading import Positions
from math import floor
from pytimeparse.timeparse import timeparse
from redis import Redis
from redis_cache import RedisCache
import alog
import binance
import click
import logging
import signal
import sys


def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier


cache = RedisCache(redis_client=Redis(host=settings.REDIS_HOST))


class TradeExecutor(BinanceUtils, Messenger):
    measurements = []
    current_position = Positions.Flat
    symbol = None
    bid_price = None

    def __init__(
        self,
        base_asset,
        futures,
        leverage,
        log_requests,
        symbol,
        quantity,
        trading_enabled,
        trade_min=False,
        fee=0.0075,
        **kwargs
    ):
        super().__init__(
            futures=futures,
            **kwargs
        )
        self._quantity = quantity
        self.trade_min = trade_min
        self.leverage = leverage
        self.symbol = f'{symbol}{base_asset}'
        self._symbol = symbol
        self.trading_enabled = trading_enabled
        self.fee = fee

        self.base_asset = base_asset
        self.asset = None
        self.ticker_channel = self.channel_suffix(f'{symbol}{self.base_asset}_book_ticker')
        self.prediction_channel = f'{symbol}{self.base_asset}_prediction'

        if leverage < self.max_leverage:
            self.client.futures_change_leverage(
                symbol=self.symbol,
                leverage=self.leverage
            )

        if log_requests:
            self.log_requests()

        self.on(self.prediction_channel, self.trade)
        self.on(self.ticker_channel, self.ticker)

    def ticker(self, data):
        self.bid_price = float(data['b'])

    @staticmethod
    @cache.cache(ttl=60 * 60)
    def leverage_brackets():
        return TradeExecutor._client().futures_leverage_bracket()

    @cached_property_with_ttl(ttl=timeparse('1h'))
    def client(self) -> Client:
        return self._client()

    @staticmethod
    def _client():
        return Client(
            api_key=settings.BINANCE_API_KEY,
            api_secret=settings.BINANCE_API_SECRET,
        )

    @property
    def asset_name(self):
        return self.symbol.replace(self.base_asset, '')

    def truncate_quantity(self, quantity):
        quantity = float(int(quantity / self.step_size) * self.step_size)

        quantity = Decimal(quantity)
        quantity = truncate(quantity, self.precision)

        return quantity

    @property
    def quantity(self):
        if self._quantity > 0.0:
            return self._quantity
        else:
            getcontext().prec = self.precision

            quantity = self.balance / (float(self.bid_price) * (1 + self.fee))

            return self.truncate_quantity(quantity)

    @property
    def min_quantity(self):
        getcontext().prec = self.precision

        quantity = 5.00 / (float(self.bid_price))

        quantity = math.ceil(quantity / self.step_size)

        quantity = Decimal(quantity)

        quantity = truncate(quantity, self.precision)

        return quantity

    @cached_property_with_ttl(ttl=2)
    def asset_quantity(self):
        getcontext().prec = self.precision

        quantity = self.client.get_asset_balance(self._symbol)['free']

        quantity = floor(Decimal(quantity) / Decimal(self.step_size)) * \
                    Decimal(self.step_size)

        quantity = Decimal(quantity)

        return quantity

    @cached_property_with_ttl(ttl=2)
    def order(self):
        orders = self.orders

        if len(self.orders) > 0:
            return orders[0]
        else:
            return None

    def trade(self, position):
        self.position = Positions(position)

        if self.bid_price is None:
            return

        alog.info('### prediction ###')
        alog.info(self.position)
        alog.info(self.quantity)

        can_trade = False

        if self.position != self.current_position:
            can_trade = True

        if can_trade:
            alog.info('## attempt trade ##')

            if len(self.orders) == 0 and self.quantity > 0  and self.position == \
                Positions.Short and self.asset_quantity < self.min_quantity:
                # add limit orders

                self.sell()

            if self.order:
                price = Decimal(self.order['price'])
                if price != self.bid_price:
                    params = dict(symbol=self.order['symbol'],
                                 orderId=self.order['orderId'])

                    alog.info(alog.pformat(params))
                    alog.infd('cancel order')

                    self.client.cancel_order(**params)

                    if self.position == Positions.Short:
                        self.sell()

            alog.info((self.position, self.asset_quantity))

            if self.position == Positions.Flat and \
                self.asset_quantity > self.tick_size:
                alog.info('### sell ###')

                self.buy(self.asset_quantity)

            self.current_position = self.position

    def sell(self):
        if self.trade_min:
            quantity = self.min_quantity
        else:
            quantity = self.quantity

        params = dict(
            symbol=self.symbol,
            side=SIDE_SELL,
            type=ORDER_TYPE_MARKET,
            quantity=quantity,
        )

        alog.info(alog.pformat(params))

        if self.trading_enabled:
            self.client.create_order(**params)

    def buy(self, quantity):
        if self.trading_enabled:
            response = self.client.create_order(
                symbol=self.symbol,
                side=SIDE_BUY,
                type=binance.enums.ORDER_TYPE_MARKET,
                quantity=quantity,
            )
            alog.info(alog.pformat(response))

    @cached_property_with_ttl(ttl=3)
    def account_data(self):
        data = dict(
            # account_status=self.client.get_account_status(),
            balance=self.client.get_asset_balance(self.base_asset),
            # account_assets=self.client.get_sub_account_assets()
            # trades=self.client.get_my_trades(symbol=self.symbol)
            # dust_log=self.client.get_dust_log(),
            # trade_fee=self.client.get_trade_fee(symbol=self.symbol)
        )

        return data

    @property
    def balance(self):
        return float(self.account_data['balance']['free'])

    @cached_property_with_ttl(ttl=timeparse('10s'))
    def orders(self):
        orders = self.client.get_all_orders(symbol=self.symbol)

        return [order for order in orders if order['status'] == 'NEW']

    def start(self):
        self.sub([self.ticker_channel,
                  self.prediction_channel])

    def stop(self):
        sys.exit(0)


@click.command()
@click.option('--base-asset', '-b', default='BNB', type=str)
@click.option('--futures', '-F', is_flag=True)
@click.option('--leverage', default=2, type=int)
@click.option('--quantity', '-q', default=0.0, type=float)
@click.option('--log-requests', '-l', is_flag=True)
@click.option('--trading-enabled', '-e', is_flag=True)
@click.option('--trade-min', is_flag=True)
@click.argument('symbol', type=str)
def main(**kwargs):
    emitter = TradeExecutor(**kwargs)
    emitter.start()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda: exit(0))
    signal.signal(signal.SIGTERM, lambda: exit(0))
    main()
