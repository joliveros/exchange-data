#!/usr/bin/env python
from math import floor

from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET
from cached_property import cached_property_with_ttl
from decimal import Decimal, getcontext
from exchange_data import settings
from exchange_data.data.measurement_frame import MeasurementFrame
from exchange_data.emitters import Messenger, binance
from exchange_data.emitters.backtest import BackTest
from exchange_data.models.resnet.study_wrapper import StudyWrapper
from exchange_data.ta_model.tune_macd import MacdParamFrame
from exchange_data.trading import Positions
from pytimeparse.timeparse import timeparse

import alog
import binance
import click
import signal
import sys


def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier


class MacdParams(object):
    def __init__(
        self,
        group_by_min=0,
        long_period=0,
        short_period=0,
        symbol=None,
        **kwargs
    ):
        self.group_by_min = group_by_min
        self.long_period = long_period
        self.short_period = short_period
        self.symbol = symbol

    def __repr__(self):
        return alog.pformat(vars(self))


class TradeExecutor(MeasurementFrame, Messenger):
    _trial_params_val = None
    measurements = []
    current_position = Positions.Flat
    symbol = None

    def __init__(
        self,
        base_asset,
        symbol,
        trading_enabled,
        depth,
        tick=False,
        fee=0.0075,
        model_version=None,
        tick_interval='5s',
        **kwargs
    ):
        super().__init__(
            **kwargs
        )
        self.depth = depth
        self.tick = tick
        self.symbol = f'{symbol}{base_asset}'
        self.trading_enabled = trading_enabled
        self.fee = fee

        self.tick_interval = tick_interval
        self.base_asset = base_asset
        self.asset = None
        self._model_version = None
        self.model_version = model_version

        if tick:
            alog.info(self.position)
            self.trade()
            sys.exit(0)

        self.on(tick_interval, self.trade)

    @cached_property_with_ttl(ttl=15)
    def client(self):
        return Client(
            api_key=settings.BINANCE_API_KEY,
            api_secret=settings.BINANCE_API_SECRET,
        )

    @cached_property_with_ttl(ttl=timeparse('10m'))
    def exchange_info(self):
        return self.client.get_exchange_info()

    @property
    def symbol_info(self):
        return [info for info in self.exchange_info['symbols'] if
                   info['symbol'] == self.symbol][0]

    @property
    def symbols(self):
        return [symbol['symbol'] for symbol in self.exchange_info['symbols']
                   if symbol['symbol'].endswith(self.base_asset)]

    @property
    def asset_name(self):
        return self.symbol.replace(self.base_asset, '')

    @property
    def lot_size(self):
        return [filter for filter in self.symbol_info['filters'] if filter[
            'filterType'] == 'LOT_SIZE'][0]

    @property
    def price_filter(self):
        return [filter for filter in self.symbol_info['filters'] if
                        filter[
            'filterType'] == 'PRICE_FILTER'][0]

    @property
    def step_size(self):
        return float(self.lot_size['stepSize'])

    @property
    def precision(self):
        return self.symbol_info['quoteAssetPrecision']

    @property
    def bid_price(self):
        getcontext().prec = self.precision

        ticker = self.client.get_orderbook_ticker(symbol=self.symbol)

        price = int((float(ticker['bidPrice'])) /
                   self.tick_size) * self.tick_size

        price = Decimal(price)

        return round(price, self.precision)

    @property
    def tick_size(self):
        return float(self.price_filter['tickSize'])

    @property
    def quantity(self):
        getcontext().prec = self.precision

        quantity = self.balance / (float(self.bid_price) * (1 + self.fee))

        quantity = float(int(quantity / self.step_size) * self.step_size)

        quantity = Decimal(quantity)

        quantity = truncate(quantity, self.precision)

        return quantity

    @property
    def min_quantity(self):
        getcontext().prec = self.precision

        quantity = 0.15 / (float(self.bid_price))

        quantity = float(int(quantity / self.step_size) * self.step_size)

        quantity = Decimal(quantity)

        quantity = truncate(quantity, self.precision)

        return quantity

    @property
    def asset_quantity(self):
        getcontext().prec = self.precision

        quantity = self.client.get_asset_balance(self.asset_name)['free']

        quantity = floor(Decimal(quantity) / Decimal(self.step_size)) * \
                    Decimal(self.step_size)

        quantity = Decimal(quantity)

        return quantity

    @cached_property_with_ttl(ttl=1)
    def order(self):
        orders = self.orders

        if len(self.orders) \
            > 0:
            return orders[0]
        else:
            return None

    def trade(self, timestamp=None):
        try:
            self._trial_params_val = self._trial_params()
        except Exception as e:
            if not self.trial_params:
                raise Exception()

        alog.info('### trade ###')
        alog.info(self.position)
        alog.info(self.quantity)

        if len(self.orders) == 0 and self.quantity > 0  and self.position == \
            Positions.Long and self.asset_quantity < self.min_quantity:
            # add limit orders

            alog.info(self.quantity)
            alog.info(self.bid_price)

            self.buy()

        if self.order:
            price = Decimal(self.order['price'])
            if price != self.bid_price:
                params = dict(symbol=self.order['symbol'],
                             orderId=self.order['orderId'])

                alog.info(alog.pformat(params))

                self.client.cancel_order(**params)

                if self.position == Positions.Long:
                    self.buy()

        if self.position == Positions.Flat and \
            self.asset_quantity > self.tick_size:
            self.sell()

    def sell(self):
        params = dict(
            symbol=self.symbol,
            side=SIDE_SELL,
            type=ORDER_TYPE_MARKET,
            quantity=self.asset_quantity,
        )

        alog.info(alog.pformat(params))

        if self.trading_enabled:
            self.client.create_order(**params)

    def buy(self):
        if self.trading_enabled:
            response = self.client.create_order(
                symbol=self.symbol,
                side=SIDE_BUY,
                type=binance.enums.ORDER_TYPE_MARKET,
                quantity=self.quantity,
            )
            alog.info(alog.pformat(response))

    def _trial_params(self):
        study = StudyWrapper(self.symbol)

        return vars(study.study.best_trial)

    @property
    def trial_params(self):
        if not self._trial_params_val:
            self._trial_params_val = self._trial_params()
        return self._trial_params_val

    @property
    def quantile(self):
        return self.trial_params['_user_attrs']['quantile']

    @property
    def model_version(self):
        if self._model_version:
            return self._model_version
        else:
            v = self.trial_params['_user_attrs']['model_version']
            self._model_version = v

        return self._model_version

    @property
    def model_params(self):
        params = self.trial_params['_params']

        if 'group_by' in params:
            group_by = params['group_by']
            params['group_by'] = f'{group_by}m'

        return params

    @model_version.setter
    def model_version(self, value):
        self._model_version = value

    @cached_property_with_ttl(ttl=48)
    def position(self) -> Positions:
        df = BackTest(
            best_exported_model=True,
            database_name=self.database_name,
            depth=self.depth,
            interval='2m',
            model_version=self.model_version,
            quantile=self.quantile,
            sequence_length=48,
            symbol=self.symbol,
            window_size='3m',
            **self.model_params
        ).test()

        return df.iloc[-1]['position']

    def best_params(self) -> MacdParams:
        params_df = MacdParamFrame(database_name=self.database_name,
                                   interval=self.interval_str)\
            .frame_all_keys()

        params = MacdParams(
            **params_df.loc[params_df['value'].idxmax()].to_dict()
        )

        return params

    @property
    def account_data(self):

        data = dict(
            account_status=self.client.get_account_status(),
            balance=self.client.get_asset_balance(self.base_asset),
            account_assets=self.client.get_sub_account_assets()
            # trades=self.client.get_my_trades(symbol=self.symbol)
            # dust_log=self.client.get_dust_log(),
            # trade_fee=self.client.get_trade_fee(symbol=self.symbol)
        )

        return data

    @property
    def balance(self):
        return float(self.account_data['balance']['free'])

    @cached_property_with_ttl(ttl=2)
    def orders(self):
        orders = self.client.get_all_orders(symbol=self.symbol)

        return [order for order in orders if order['status'] == 'NEW']

    def start(self):
        self.sub([self.tick_interval])

    def stop(self):
        sys.exit(0)


@click.command()
@click.option('--database-name', '-d', default='binance', type=str)
@click.option('--base-asset', '-b', default='BNB', type=str)
@click.option('--tick-interval', '-t', default='1m', type=str)
@click.option('--interval', '-i', default='2m', type=str)
@click.option('--depth', default=72, type=int)
@click.option('--model-version', '-m', default=None, type=str)
@click.option('--trading-enabled', '-e', is_flag=True)
@click.option('--tick', is_flag=True)
@click.argument('symbol', type=str)
def main(**kwargs):
    emitter = TradeExecutor(**kwargs)
    emitter.start()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda: exit(0))
    signal.signal(signal.SIGTERM, lambda: exit(0))
    main()
