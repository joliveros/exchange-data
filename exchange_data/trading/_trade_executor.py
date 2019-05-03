import json
from datetime import timedelta
from time import sleep

import alog
from bitmex import bitmex

from exchange_data import settings, Database
from exchange_data.channels import BitmexChannels
from exchange_data.emitters import SignalInterceptor
from exchange_data.emitters.messenger import Messenger
from exchange_data.utils import DateTimeUtils
from tgym.envs.orderbook.utils import Positions

import click


class TradeExecutor(
    Database,
    Messenger,
    DateTimeUtils,
    SignalInterceptor
):

    def __init__(
        self,
        algo_channel: str,
        symbol: str,
        position_size: int = 1,
        **kwargs):
        Database.__init__(self, database_name='bitmex', **kwargs)
        SignalInterceptor.__init__(self, exit_func=self.stop, **kwargs)
        Messenger.__init__(self, **kwargs)
        DateTimeUtils.__init__(self)

        self.symbol = BitmexChannels[symbol]
        self.algo_channel = algo_channel

        self.last_position = self._last_position()
        self.position_size = position_size
        self.bitmex_client = bitmex(
            test=False,
            api_key=settings.BITMEX_API_KEY,
            api_secret=settings.BITMEX_API_SECRET
        )

        self.on(self.algo_channel, self.execute)

    def _last_position(self):
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

    def start(self):
        self.sub([self.algo_channel])

    def execute(self, action):
        position = Positions[action['data']]

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
    'algo_channel',
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
