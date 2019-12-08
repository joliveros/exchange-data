#!/usr/bin/env python

from abc import ABC
from datetime import timedelta
from exchange_data.streamers._bitmex import BitmexStreamer
from exchange_data.utils import DateTimeUtils
from pytimeparse.timeparse import timeparse

import alog
import click
import json
import numpy as np
import tensorflow as tf

CLASSES = [0, 1, 2]
# table = index_table_from_tensor(tf.constant(CLASSES), dtype=int64)
NUM_CLASSES = len(CLASSES)


class OrderBookImgStreamer(BitmexStreamer, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = []

    def orderbook_frame_query(self, start_date=None, end_date=None):
        start_date = start_date if start_date else self.start_date
        end_date = end_date if end_date else self.end_date

        start_date = self.format_date_query(start_date)
        end_date = self.format_date_query(end_date)
        query = f'SELECT FIRST(*) as data FROM {self.channel_name} ' \
            f'WHERE time >= {start_date} AND time <= {end_date} ' \
            f'GROUP BY time({self.sample_interval});'

        alog.info(query)

        return self.query(
            query,
            chunk_size=4112
        )

    def _orderbook_frames(self, start_date, end_date):
        orderbook = self.orderbook_frame_query(start_date, end_date)

        return orderbook.get_points(self.channel_name)

    def orderbook_frames(self):
        diff = self.start_date - self.end_date
        diff2 = self.start_date - self.original_end_date

        if diff == diff2:
            self.end_date = self.start_date + self.window_delta

        return self._orderbook_frames(self.start_date, self.end_date)

    def next_window(self):
        result = self.orderbook_frames()

        self._set_next_window()

        return result

    def send(self, *args):
        self.counter += 1

        if len(self.data) == 0:
            self.data = [frame for frame in self.next_window()]
            self.data.pop(0)

        return self.data.pop(0)


def data_streamer(interval: str = '15s', **kwargs):
    end_date = DateTimeUtils.now()
    start_date = end_date - timedelta(seconds=timeparse(interval))

    streamer = OrderBookImgStreamer(
        start_date=start_date,
        end_date=end_date,
        window_size='2s',
        channel_name='orderbook_img_frame_XBTUSD',
        **kwargs
    )

    for data in streamer:
        time = data['time']
        expected_position = data['data_expected_position']
        frame = np.array(json.loads(data['data_frame']))

        alog.info((time, expected_position, frame.shape))

        yield data

def dataset(batch_size: int, epochs: int = 1, **kwargs):
    return data_streamer(**kwargs)

    #
    # tf.data.Dataset.from_generator(
    #     streamer,
    #     output_types=(tf.int32, tf.float32),
    #     output_shapes=((), (None,))
    # )
    #
    # _dataset = _dataset.map(extract_fn)\
    #     .batch(batch_size) \
    #     .repeat(epochs)
    #
    # return _dataset


@click.command()
# @click.option('--summary-interval', '-s', default=6, type=int)
@click.option('--interval', '-i', default='15s', type=str)
def main(**kwargs):
    for x in dataset(2, **kwargs):
        # alog.info(x['time'])
        pass


if __name__ == '__main__':
    main()
