import re
from datetime import timedelta, datetime
from pathlib import Path

from dateutil import parser
from pytimeparse.timeparse import timeparse
from tensorflow.core.example.example_pb2 import Example
from tensorflow.core.example.feature_pb2 import Features, FloatList, \
    BytesList, Feature, Int64List
from tensorflow.python.lib.io.tf_record import TFRecordWriter, TFRecordCompressionType

from exchange_data.utils import DateTimeUtils
from tgym.envs import OrderBookTradingEnv
from tgym.envs.orderbook.utils import Positions

from time import sleep
import alog
import click
import random


class OrderBookTFRecord(OrderBookTradingEnv):
    def __init__(self, filename, **kwargs):
        OrderBookTradingEnv.__init__(
            self,
            random_start_date=False,
            use_volatile_ranges=False,
            **kwargs
        )

        self.reset()

        self.file_path = f'{Path.home()}/.exchange-data/tfrecords/' \
            f'{filename}.tfrecord'

    def reset(self, **kwargs):
        if self.step_count > 0:
            alog.debug('##### reset ######')
            alog.info(alog.pformat(self.summary()))

        _kwargs = self._args['kwargs']
        del self._args['kwargs']
        _kwargs = {**self._args, **_kwargs, **kwargs}

        new_instance = OrderBookTradingEnv(**_kwargs)
        self.__dict__ = new_instance.__dict__

        for i in range(self.max_frames):
            self.get_observation()

        return self.last_observation

    @property
    def avg_exit_price(self):
        return (self.best_ask + self.best_bid) / 2

    @property
    def avg_entry_price(self):
        return (self.last_best_ask + self.last_best_bid) / 2

    @property
    def expected_position(self):
        position = None
        diff = self.avg_exit_price - self.avg_entry_price

        if diff > 0.0:
            position = Positions.Long
        elif diff < 0.0:
            position = Positions.Short
        elif diff == 0.0:
            position = Positions.Flat

        return position

    def run(self):
        now = self.now()

        with TFRecordWriter(self.file_path, TFRecordCompressionType.GZIP) as writer:
            while self._last_datetime < now:
                self.write_observation(writer)

    def write_observation(self, writer):
        self.get_observation()
        self.step_count += 1
        # alog.info(self.frames[-2].shape)
        # raise Exception()

        data = dict(
            datetime=self.BytesFeature(self.last_datetime),
            frame=self.floatFeature(self.frames[-2].flatten()),
            expected_position=self.int64Feature(self.expected_position.value),
        )
        example: Example = Example(features=Features(feature=data))
        writer.write(example.SerializeToString())

    def int64Feature(self, value):
        return Feature(int64_list=Int64List(value=[value]))

    def floatFeature(self, value):
        return Feature(float_list=FloatList(value=value))

    def BytesFeature(self, value):
        return Feature(bytes_list=BytesList(value=[bytes(value, encoding='utf8')]))


@click.command()
@click.option('--summary-interval', '-s', default=6, type=int)
@click.option('--interval', '-i', default='1h', type=str)
@click.option('--max-frames', '-m', default=12, type=int)
@click.option('--print-ascii-chart', '-a', is_flag=True)
def main(interval, **kwargs):
    start_date = DateTimeUtils.now() - timedelta(seconds=timeparse(interval))
    filename = re.sub('[:+\s\-]', '_', str(start_date).split('.')[0])

    record = OrderBookTFRecord(
        window_size='1m',
        is_training=False,
        frame_width=96,
        start_date=start_date,
        filename=filename,
        **kwargs
    )

    record.run()


if __name__ == '__main__':
    main()

