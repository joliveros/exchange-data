#!/usr/bin/env python

from exchange_data import Measurement, NumpyEncoder, settings
from exchange_data.channels import BitmexChannels
from exchange_data.emitters import Messenger
from exchange_data.trading import Positions
from exchange_data.utils import DateTimeUtils
from tgym.envs.orderbook._orderbook import OrderBookIncompleteException, \
    OrderBookTradingEnv

from tgym.envs.orderbook.ascii_image import AsciiImage

import alog
import click
import json
import logging
import numpy as np


class TrainingDataBase(object):
    def __init__(self, min_change=0.0, **kwargs):
        self.min_change = min_change
        self.last_best_bid = None
        self.last_best_ask = None
        self.best_bid = None
        self.best_ask = None

        super().__init__()

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
        abs_diff = abs(diff)

        if diff > 0.0 and abs_diff >= self.min_change:
            position = Positions.Long
        elif diff < 0.0 and abs_diff >= self.min_change:
            position = Positions.Short
        else:
            position = Positions.Flat

        return position


class OrderBookTrainingData(Messenger, OrderBookTradingEnv, TrainingDataBase):
    def __init__(
        self,
        depth=21,
        symbol=BitmexChannels.XBTUSD,
        **kwargs
    ):
        start_date = DateTimeUtils.now()

        super().__init__(
            database_batch_size=1,
            database_name='bitmex',
            date_checks=False,
            end_date=start_date,
            orderbook_depth=depth,
            random_start_date=False,
            start_date=start_date,
            **kwargs
        )

        self.symbol = symbol
        self._last_datetime = self.start_date
        self.last_data = []
        self.orderbook_channel = f'XBTUSD_OrderBookFrame_depth_{depth}'
        self.channel_name = f'orderbook_img_frame_{self.symbol.value}_{depth}'

        self.on(self.orderbook_channel, self.write_observation)
        self.on('frame_str', self.publish_to_channels)
        self.on('frame_str', self.write_to_db)

    def _get_observation(self):
        time = self.last_data['time']
        orderbook = json.loads(self.last_data['fields']['data'])
        orderbook = np.array(orderbook)

        self.last_timestamp = time
        return time, orderbook

    def write_observation(self, data):
        self.last_data = data

        self.get_observation()

        if len(self.frames) >= self.max_frames:
            frame = self.frames[-2]

            if self.expected_position != Positions.Flat:
                alog.info((self.expected_position, self.diff, self.best_ask,
                           self.best_bid))

            # if settings.LOG_LEVEL == logging.DEBUG:
            # alog.info(AsciiImage(frame, new_width=21))

            frame_str = json.dumps(frame, cls=NumpyEncoder)

            self.emit('frame_str', frame_str)

    def publish_to_channels(self, frame_str):
        self.publish(self.channel_name, json.dumps(dict(
            frame=frame_str
        )))

    def write_to_db(self, frame_str):
        timestamp = DateTimeUtils.parse_datetime_str(self.last_timestamp)

        measurement = Measurement(
            measurement=self.channel_name,
            time=timestamp,
            tags=dict(symbol=self.symbol.value),
            fields=dict(
                frame=frame_str,
                expected_position=self.expected_position.value,
                entry_price=self.avg_entry_price,
                best_ask=self.best_ask,
                best_bid=self.best_bid
            )
        )

        super().write_points([measurement.__dict__])

    def run(self):
        self.sub([self.orderbook_channel, BitmexChannels.XBTUSD])


@click.command()
@click.option('--frame-width', default=96, type=int)
@click.option('--min-change', default=2.0, type=float)
@click.option('--min-std-dev', '-std', default=2.0, type=float)
@click.option('--top-limit', '-l', default=5e5, type=float)
@click.option('--print-ascii-chart', '-a', is_flag=True)
@click.option('--summary-interval', '-si', default=6, type=int)
@click.option('--max-frames', '-m', default=6, type=int)
@click.option('--depth', '-d', default=21, type=int)
@click.option('--use-volatile-ranges', '-v', is_flag=True)
def main(**kwargs):
    record = OrderBookTrainingData(**kwargs)
    record.run()


if __name__ == '__main__':
    main()
