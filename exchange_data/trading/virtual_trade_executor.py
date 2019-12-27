#! /usr/bin/env python
import json

from exchange_data.bitmex_orderbook import BitmexOrderBook
from exchange_data.channels import BitmexChannels
from exchange_data.emitters import Messenger, TimeChannels
from exchange_data.emitters.prediction_emitter import TradeJob
from exchange_data.trading import Positions
from exchange_data.trading._trade_executor import TradeExecutorUtil
from tgym.envs import OrderBookTradingEnv

import alog
import click
import random
import numpy as np

class VirtualTradeExecutor(
    OrderBookTradingEnv,
    Messenger,
    TradeExecutorUtil,
    TradeJob
):
    def __init__(
        self,
        symbol,
        **kwargs
    ):
        TradeJob.__init__(self, symbol=symbol)

        super().__init__(
            symbol=symbol,
            database_name='bitmex',
            random_start_date=True,
            # exit_func=self.stop,
            **kwargs
        )

        self._last_position = Positions.Flat
        self.orderbook_frame_channel = 'XBTUSD_OrderBookFrame_depth_21'
        self.last_frame = None
        self.on(self.orderbook_frame_channel, self.handle_frame)
        self.on(self.job_name, self.execute)
        # self.on(TimeChannels.Tick.value, self.execute)

    @property
    def last_position(self):
        return self._last_position

    @last_position.setter
    def last_position(self, value):
        assert type(value) == Positions
        self._last_position = value

    def start(self, channels=[]):
        self.sub(channels + [self.job_name, self.orderbook_frame_channel])

    def handle_frame(self, frame):
        self.last_frame = (
            self.parse_datetime_str(frame['time']),
            np.array(json.loads(frame['fields']['data']))
        )

    def _get_observation(self):
        return self.last_frame

    def execute(self, action):
        if self.last_frame:
            self._execute(action)

    def _execute(self, action):
        position = self.parse_position_value(int(action['data']))
        # position = self.parse_position_value(random.randint(0, 2))

        if len(self.position_history) == 0:
            self.get_observation()

        self.get_observation()

        self.step_position(position.value)

        alog.info(alog.pformat(self.summary()))

@click.command()
@click.option('--position-size', '-p', type=int, default=1)
@click.argument('symbol', type=click.Choice(BitmexChannels.__members__))
def main(symbol, **kwargs):
    executor = VirtualTradeExecutor(
        symbol=BitmexChannels[symbol],
        capital=1.0,
        max_summary=10,
        **kwargs)

    executor.start()

if __name__ == '__main__':
    main()
