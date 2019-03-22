import alog
from bitmex import bitmex

from exchange_data import settings
from exchange_data.channels import BitmexChannels
from exchange_data.emitters import SignalInterceptor
from exchange_data.emitters.messenger import Messenger
from exchange_data.utils import DateTimeUtils
from tgym.envs.orderbook.utils import Positions

import click


class TradeExecutor(
    Messenger,
    DateTimeUtils,
    SignalInterceptor
):

    def __init__(self, algo_channel: str, symbol: str, **kwargs):
        super().__init__(exit_func=self.stop, **kwargs)
        self.symbol = BitmexChannels[symbol]
        self.algo_channel = algo_channel
        self.last_position = None
        self.position_size = 1
        self.bitmex_client = bitmex(
            test=False,
            api_key=settings.BITMEX_API_KEY,
            api_secret=settings.BITMEX_API_SECRET
        )
        self.on(self.algo_channel, self.execute)

    def start(self):
        self.sub([self.algo_channel])

    def execute(self, action):
        position = Positions[action['data']]
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

        result = self.bitmex_client.Order.Order_new(
            symbol=self.symbol.value,
            orderQty=self.position_size,
            side=side,
            ordType='Market'
        ).result()[0]

        alog.info(result)

    def long(self):
        result = self.bitmex_client.Order.Order_new(
            symbol=self.symbol.value,
            ordType='Market',
            orderQty=self.position_size,
            side='Buy'
        ).result()[0]
        alog.info(result)

    def short(self):
        result = self.bitmex_client.Order.Order_new(
            symbol=self.symbol.value,
            ordType='Market',
            orderQty=self.position_size,
            side='Sell'
        ).result()[0]
        alog.info(result)

@click.command()
@click.argument(
    'algo_channel',
    type=str,
    default=None
)
@click.argument('symbol', type=click.Choice(BitmexChannels.__members__))
def main(**kwargs):
    time_emitter = TradeExecutor(**kwargs)
    time_emitter.start()


if __name__ == '__main__':
    main()
