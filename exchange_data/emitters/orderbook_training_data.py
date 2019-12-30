#!/usr/bin/env python

from exchange_data import Measurement, NumpyEncoder
from exchange_data.channels import BitmexChannels
from exchange_data.emitters import Messenger
from exchange_data.trading import Positions
from exchange_data.utils import DateTimeUtils, Base
from tgym.envs import OrderBookTradingEnv
from tgym.envs.orderbook._orderbook import OrderBookIncompleteException
from tgym.envs.orderbook.ascii_image import AsciiImage

import alog
import click
import json
import numpy as np


class TrainingDataBase(object):
    def __init__(self):
        self.last_best_bid = None
        self.last_best_ask = None
        self.best_bid = None
        self.best_ask = None

    @property
    def avg_exit_price(self):
        return (self.best_ask + self.best_bid) / 2

    @property
    def avg_entry_price(self):
        return (self.last_best_ask + self.last_best_bid) / 2

    @property
    def diff(self):
        return self.avg_exit_price - self.avg_entry_price

    @property
    def expected_position(self):
        position = None
        diff = self.diff

        if diff > 0.0:
            position = Positions.Long
        elif diff < 0.0:
            position = Positions.Short
        elif diff == 0.0:
            position = Positions.Flat

        # alog.debug((diff, position))

        return position


class OrderBookTrainingData(Messenger, OrderBookTradingEnv, TrainingDataBase):
    def __init__(
        self,
        symbol=BitmexChannels.XBTUSD,
        **kwargs
    ):
        start_date = DateTimeUtils.now()

        super().__init__(
            start_date=start_date,
            database_name='bitmex',
            random_start_date=False,
            database_batch_size=3,
            date_checks=False,
            **kwargs
        )

        self.symbol = symbol
        self._last_datetime = self.start_date
        self.last_data = []
        self.orderbook_channel = 'XBTUSD_OrderBookFrame_depth_21'

        self.on(self.orderbook_channel, self.write_observation)

    def _get_observation(self):
        time = self.last_data['time']
        orderbook = json.loads(self.last_data['fields']['data'])
        orderbook = np.array(orderbook)

        self.last_timestamp = time
        return time, orderbook

    def write_observation(self, data):
        self.last_data = data

        try:
            self.get_observation()

        except OrderBookIncompleteException as e:
            pass

        if len(self.frames) >= self.max_frames:
            frame = self.frames[-2]

            frame_str = json.dumps(frame, cls=NumpyEncoder)

            channel_name = f'orderbook_img_frame_{self.symbol.value}'

            timestamp = DateTimeUtils.parse_datetime_str(self.last_timestamp)

            if self.expected_position != Positions.Flat:
                alog.info((self.expected_position, self.best_ask, self.best_bid))

            measurement = Measurement(
                measurement=channel_name,
                time=timestamp,
                tags=dict(symbol=self.symbol.value),
                fields=dict(
                    frame=frame_str,
                    expected_position=self.expected_position.value,
                    entry_price=self.avg_entry_price
                )
            )

            self.publish(channel_name, json.dumps(dict(
                frame=frame_str
            )))

            super().write_points([measurement.__dict__])

    def run(self):
        self.sub([self.orderbook_channel, BitmexChannels.XBTUSD])


@click.command()
@click.option('--frame-width', default=96, type=int)
@click.option('--min-std-dev', '-std', default=2.0, type=float)
@click.option('--print-ascii-chart', '-a', is_flag=True)
@click.option('--summary-interval', '-si', default=6, type=int)
@click.option('--max-frames', '-m', default=6, type=int)
@click.option('--use-volatile-ranges', '-v', is_flag=True)
def main(**kwargs):
    record = OrderBookTrainingData(**kwargs)
    record.run()


if __name__ == '__main__':
    main()
