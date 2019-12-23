#!/usr/bin/env python

from abc import ABC
from datetime import timedelta
from exchange_data.streamers._bitmex import BitmexStreamer
from exchange_data.trading import Positions
from exchange_data.utils import DateTimeUtils
from pytimeparse.timeparse import timeparse

import alog
import click
import json
import numpy as np
import tensorflow as tf

from tgym.envs.orderbook.ascii_image import AsciiImage

CLASSES = [0, 1, 2]
# table = index_table_from_tensor(tf.constant(CLASSES), dtype=int64)
NUM_CLASSES = len(CLASSES)


class OrderBookImgStreamer(BitmexStreamer, ABC):
    def __init__(
        self,
        side,
        steps_epoch: str,
        use_volatile_ranges: bool = False,
        **kwargs
    ):
        super().__init__(database_name='bitmex', **kwargs)
        self.side = side
        self.data = []
        self.steps_epoch = timeparse(steps_epoch)
        self.use_volatile_ranges = use_volatile_ranges
        self.current_range = None

        if use_volatile_ranges:
            self.volatile_ranges = self.get_volatile_ranges(side)

    def get_volatile_ranges(self, side, start_date=None, end_date=None):
        start_date = start_date if start_date else self.start_date
        end_date = end_date if end_date else self.end_date

        start_date = self.format_date_query(start_date)
        end_date = self.format_date_query(end_date)

        query = f'SELECT e FROM (SELECT expected_position as e ' \
            f'from {self.channel_name} ' \
            f'WHERE time >= {start_date} AND time <= {end_date}) ' \
            f'WHERE e = {side};'

        alog.info(query)

        ranges = self.query(query).get_points(self.channel_name)
        timestamps = [data['time'] for data in ranges]

        return [(
               DateTimeUtils.parse_db_timestamp(timestamp) - timedelta(seconds=3),
               DateTimeUtils.parse_db_timestamp(timestamp) + timedelta(seconds=1)
            ) for timestamp in timestamps]

    def orderbook_frame_query(self, start_date=None, end_date=None):
        start_date = start_date if start_date else self.start_date
        end_date = end_date if end_date else self.end_date

        start_date = self.format_date_query(start_date)
        end_date = self.format_date_query(end_date)
        query = f'SELECT FIRST(*) as data FROM {self.channel_name} ' \
            f'WHERE time >= {start_date} AND time <= {end_date} ' \
            f'GROUP BY time({self.sample_interval});'

        alog.info(query)

        return self.query(query)

    def _orderbook_frames(self, start_date, end_date):
        orderbook = self.orderbook_frame_query(start_date, end_date)

        return orderbook.get_points(self.channel_name)

    def orderbook_frames(self):
        return self._orderbook_frames(self.start_date, self.end_date)

    def _set_next_window(self):
        if self.use_volatile_ranges and self.side != 0:
            if self.current_range is None:
                if len(self.volatile_ranges) == 0:
                    raise StopIteration()

                self.current_range = self.volatile_ranges.pop(0)
                alog.info('#### new range ###')
                self.start_date = self.current_range[0]
            else:
                self.start_date += self.window_delta

            self.end_date = self.start_date + self.window_delta

            now = DateTimeUtils.now()

            alog.info((str(self.end_date), str(now)))

            if self.end_date > now:
                self.end_date = now

            if self.end_date >= self.current_range[1]:
                self.current_range = None
        else:
            if self.counter > 0:
                self.start_date += timedelta(seconds=self.window_size)
            self.end_date = self.start_date + self.window_delta

        if self.end_date >= self.stop_date:
            raise StopIteration()

    def next_window(self):
        self._set_next_window()

        result = self.orderbook_frames()

        return result

    def send(self, *args):
        if len(self.data) == 0:
            self.data = [frame for frame in self.next_window()]

            if len(self.data) == 0:
                raise StopIteration()

            self.data.pop(0)

        if self.counter >= self.steps_epoch:
            raise StopIteration()

        self.counter += 1

        data = self.data.pop(0)
        return data


def data_streamer(frame_width, side, interval: str = '15s', **kwargs):
    end_date = DateTimeUtils.now()
    start_date = end_date - timedelta(seconds=timeparse(interval))

    streamer = OrderBookImgStreamer(
        start_date=start_date,
        end_date=end_date,
        channel_name='orderbook_img_frame_XBTUSD',
        side=side,
        **kwargs
    )

    for data in streamer:
        expected_position = data['data_expected_position']

        frame = np.array(json.loads(data['data_frame']), dtype=np.uint8)

        if side != 0:
            expected_position = side

        alog.info(AsciiImage(frame, new_width=10))
        alog.info(expected_position)

        yield frame, expected_position


def _dataset(frame_width, batch_size: int, epochs: int = 1, **kwargs):
    kwargs['frame_width'] = frame_width

    return tf.data.Dataset.from_generator(
        lambda: data_streamer(**kwargs),
        output_types=(tf.float32, tf.int32),
        output_shapes=((frame_width, frame_width, 3,), ())
    ) \
        .batch(batch_size) \
        .repeat(epochs)

def dataset(**kwargs):
    return _dataset(side=1, **kwargs)\
        .concatenate(_dataset(side=2, **kwargs))\
        .concatenate(_dataset(side=0, **kwargs))



@click.command()
# @click.option('--summary-interval', '-s', default=6, type=int)
@click.option('--frame-width', '-f', default=225, type=int)
@click.option('--interval', '-i', default='15s', type=str)
@click.option('--steps-epoch', '-s', default='1m', type=str)
@click.option('--use-volatile-ranges', '-v', is_flag=True)
@click.option('--window-size', '-w', default='3s', type=str)
def main(**kwargs):
    for frame, expected_position in dataset(batch_size=2, **kwargs):
        alog.info((expected_position, frame.shape))
        alog.info(expected_position.numpy())



if __name__ == '__main__':
    main()
