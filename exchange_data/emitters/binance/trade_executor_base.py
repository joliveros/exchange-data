from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL
from cached_property import cached_property_with_ttl
from decimal import Decimal, getcontext
from exchange_data import settings
from exchange_data.emitters import Messenger
from exchange_data.emitters import binance
from exchange_data.emitters.SlackEmitter import SlackEmitter
from exchange_data.emitters.binance import BinanceUtils
from exchange_data.emitters.binance import truncate
from exchange_data.trading import Positions
from math import floor
from pytimeparse.timeparse import timeparse

import alog
import binance
import math
import sys


class TradeExecutorBase(BinanceUtils, SlackEmitter, Messenger):
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
        once,
        trade_min=False,
        fee=0.0075,
        **kwargs,
    ):
        super().__init__(futures=futures, channel="trades", **kwargs)

        self.once = once
        self._quantity = quantity
        self.trade_min = trade_min
        self.leverage = leverage
        self.symbol = f"{symbol}{base_asset}"
        self._symbol = symbol
        self.trading_enabled = trading_enabled
        self.short_enabled = True
        self.fee = fee
        self.position = None
        self.base_asset = base_asset
        self.asset = None
        self.ticker_channel = self.channel_suffix(
            f"{symbol}{self.base_asset}_book_ticker"
        )
        self.prediction_channel = f"{symbol}{self.base_asset}_prediction"

        info = self.exchange_info

        if leverage < self.max_leverage:
            self.client.futures_change_leverage(
                symbol=self.symbol, leverage=self.leverage
            )

        if log_requests:
            self.log_requests()

        self.on(self.ticker_channel, self.ticker)

        if not once:
            self.on(self.prediction_channel, self.trade)

        self.message("### starting up trade executor ###")

    def ticker(self, data):
        self.bid_price = float(data["b"])

        if self.once:
            self.once = False
            self.trade(1)
            self.stop()

    @staticmethod
    @cache.cache(ttl=60 * 60)
    def leverage_brackets():
        return TradeExecutor._client().futures_leverage_bracket()

    @cached_property_with_ttl(ttl=timeparse("1m"))
    def client(self) -> Client:
        client = self._client()
        return client

    @staticmethod
    def _client():
        return Client(
            api_key=settings.BINANCE_API_KEY,
            api_secret=settings.BINANCE_API_SECRET,
        )

    @property
    def asset_name(self):
        return self.symbol.replace(self.base_asset, "")

    def truncate_quantity(self, quantity):
        quantity = float(int(quantity / self.step_size) * self.step_size)

        quantity = Decimal(quantity)
        quantity = truncate(quantity, self.precision)

        return quantity

    @property
    def quantity(self):
        getcontext().prec = self.precision

        quantity = self.balance / (float(self.bid_price) * (1 + self.fee))

        return self.truncate_quantity(quantity * self.leverage)

    @property
    def min_quantity(self):
        getcontext().prec = self.precision

        quantity = (5.00 * self.leverage) / (float(self.bid_price))

        quantity = math.ceil(quantity / self.step_size)

        quantity = Decimal(quantity)

        quantity = truncate(quantity, self.precision)

        return quantity

    @cached_property_with_ttl(ttl=2)
    def asset_quantity(self):
        getcontext().prec = self.precision

        quantity = self.client.get_asset_balance(self._symbol)["free"]

        quantity = floor(Decimal(quantity) / Decimal(self.step_size)) * Decimal(
            self.step_size
        )

        quantity = Decimal(quantity)

        return quantity

    @cached_property_with_ttl(ttl=2)
    def order(self):
        orders = self.orders

        if len(self.orders) > 0:
            return orders[0]
        else:
            return None

    @property
    def exchange_position(self):
        position = [
            pos
            for pos in self.client.futures_position_information()
            if pos["symbol"] == self.symbol
        ][0]

        return position

    @property
    def positionAmt(self):
        return float(self.exchange_position["positionAmt"])

    def trade(self, position):
        position = Positions(position)

        if self.position != position:
            self.short_enabled = True

        self.position = position

        if self.bid_price is None:
            return

        alog.info("### begin trade step ###")
        alog.info(self.position)
        alog.info(self.quantity)

        can_trade = False
        positionAmt = self.positionAmt

        if self.position != self.current_position or positionAmt != 0.0:
            can_trade = True

        if can_trade:
            self._trade(positionAmt)

    def _trade(self, position_amt):
        alog.info("## attempt trade ##")
        if (
            len(self.orders) == 0
            and self.quantity > 0
            and self.position == Positions.Short
            and self.asset_quantity < self.min_quantity
        ):
            # add limit orders
            if position_amt == 0.0:
                self.sell()

        if self.order:
            alog.info("## order exists ##")

            price = Decimal(self.order["price"])

            if price != self.bid_price:
                params = dict(
                    symbol=self.order["symbol"], orderId=self.order["orderId"]
                )

                alog.info(alog.pformat(params))
                alog.infd("cancel order")

                self.client.cancel_order(**params)

                position_amt = self.positionAmt

                if self.position == Positions.Short and position_amt == 0.0:
                    self.sell()

        position_amt = self.positionAmt
        alog.info((self.position, self.asset_quantity))
        alog.info(position_amt)
        if self.position == Positions.Flat and position_amt < 0.0:
            self.buy(abs(position_amt))
        self.current_position = self.position

    def sell(self):
        if self.trade_min:
            quantity = self.min_quantity
        else:
            if self._quantity == 0:
                quantity = self.quantity
            else:
                quantity = self._quantity

        params = dict(
            symbol=self.symbol,
            side=SIDE_SELL,
            type=binance.enums.FUTURE_ORDER_TYPE_MARKET,
            quantity=quantity,
        )

        alog.info(alog.pformat(params))

        if self.trading_enabled and self.short_enabled:
            response = self.client.futures_create_order(**params)

            alog.info(alog.pformat(response))
            self.message(alog.pformat(response))

        self.short_stop_loss()

        self.short_enabled = False

    def short_stop_loss(self):
        params = dict(
            symbol=self.symbol,
            side=SIDE_BUY,
            type=binance.enums.FUTURE_ORDER_TYPE_STOP_MARKET,
            closePosition=True,
            stopPrice=round(self.bid_price * 1.005, 3),
        )

        alog.info(alog.pformat(params))

        if self.trading_enabled and self.short_enabled:
            response = self.client.futures_create_order(**params)
            alog.info(alog.pformat(response))

    def buy(self, quantity):
        if self.trading_enabled:
            response = self.client.futures_create_order(
                symbol=self.symbol,
                side=SIDE_BUY,
                type=binance.enums.ORDER_TYPE_MARKET,
                quantity=quantity,
            )
            alog.info(alog.pformat(response))
            self.message(alog.pformat(response))

        client: Client = self.client

        client.futures_cancel_all_open_orders(symbol=self.symbol)

    @cached_property_with_ttl(ttl=3)
    def account_data(self):
        data = dict(
            # account_status=self.client.get_account_status(),
            balance=self.client.futures_account_balance(),
            # account_assets=self.client.get_sub_account_assets()
            # trades=self.client.get_my_trades(symbol=self.symbol)
            # dust_log=self.client.get_dust_log(),
            # trade_fee=self.client.get_trade_fee(symbol=self.symbol)
        )

        return data

    @property
    def balance(self):
        balance = [
            assetBalance
            for assetBalance in self.account_data["balance"]
            if assetBalance["asset"] == self.base_asset
        ][0]

        return float(balance["balance"])

    @cached_property_with_ttl(ttl=timeparse("10s"))
    def orders(self):
        orders = self.client.get_all_orders(symbol=self.symbol)

        return [order for order in orders if order["status"] == "NEW"]

    def start(self):
        self.sub([self.ticker_channel, self.prediction_channel])

    def stop(self):
        sys.exit(0)
