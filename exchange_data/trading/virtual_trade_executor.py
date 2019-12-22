#! /usr/bin/env python

from bitmex import bitmex
from datetime import timedelta
from exchange_data import settings
from exchange_data.bitmex_orderbook import BitmexOrderBook
from exchange_data.channels import BitmexChannels
from exchange_data.trading import Positions
from exchange_data.trading._trade_executor import TradeExecutor

import alog
import click


class VirtualTradeExecutor(BitmexOrderBook, TradeExecutor):

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(database_name='bitmex', exit_func=self.stop, **kwargs)
        self._last_position = Positions.Flat

        self.on(self.symbol.value, self.message)

    @property
    def last_position(self):
        return self._last_position

    @last_position.setter
    def last_position(self, value):
        assert type(value) == Positions
        self._last_position = value

    def start(self, channels=[]):
        super().start(channels + [self.symbol])

    def execute(self, action):
        raise Exception()
        position = self.parse_position_value(int(action['data']))

        alog.info(position)

        if position.value != self.last_position.value:
            if position == Positions.Flat:
                self.close()
            elif position == Positions.Long:
                self.long()
            elif position == Positions.Short:
                self.short()

            self.last_position = position

    def close(self):
        side = None
        if self.last_position == Positions.Long:
            side = 'Sell'
        elif self.last_position == Positions.Short:
            side = 'Buy'
        elif self.last_position == Positions.Flat:
            return

        # result = self.bitmex_client.Order.Order_new(
        #     symbol=self.symbol.value,
        #     orderQty=self.position_size,
        #     side=side,
        #     ordType='Market'
        # ).result()[0]
        #
        # alog.info(alog.pformat(result))

    def long(self):
        # result = self.bitmex_client.Order.Order_new(
        #     symbol=self.symbol.value,
        #     ordType='Market',
        #     orderQty=self.position_size,
        #     side='Buy'
        # ).result()[0]
        # alog.info(alog.pformat(result))
        pass

    def short(self):
        # result = self.bitmex_client.Order.Order_new(
        #     symbol=self.symbol.value,
        #     ordType='Market',
        #     orderQty=self.position_size,
        #     side='Sell'
        # ).result()[0]
        # alog.info(alog.pformat(result))
        pass


@click.command()
@click.option('--position-size', '-p', type=int, default=1)
@click.argument('symbol', type=click.Choice(BitmexChannels.__members__))
def main(symbol, **kwargs):
    executor = VirtualTradeExecutor(symbol=BitmexChannels[symbol], **kwargs)
    executor.start()


if __name__ == '__main__':
    main()
