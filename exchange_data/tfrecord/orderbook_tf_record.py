#!/usr/bin/env python

from exchange_data.emitters import Messenger
from exchange_data.tfrecord.date_range_split_workers import DateRangeSplitWorkers
from exchange_data.trading import Positions
from exchange_data.utils import DateTimeUtils
from pathlib import Path
from pytimeparse.timeparse import timeparse
from tensorflow.core.example.feature_pb2 import FloatList, BytesList, Feature, Int64List
from tensorflow.python.lib.io.tf_record import TFRecordWriter, TFRecordCompressionType
from tgym.envs import OrderBookTradingEnv
from time import sleep

import alog
import click
import re
import shutil
import tensorflow as tf

Features = tf.train.Features
Example = tf.train.Example


class OrderBookTFRecord(OrderBookTradingEnv):
    def __init__(
        self,
        directory_name=None,
        directory=None,
        start_date=None,
        end_date=None,
        **kwargs
    ):
        super().__init__(
            random_start_date=False,
            use_volatile_ranges=False,
            start_date=start_date,
            end_date=end_date,
            **kwargs
        )

        filename = re.sub('[:+\s\-]', '_', str(start_date).split('.')[0])

        if directory_name is None:
            raise Exception()

        if directory is None:
            directory = f'{Path.home()}/.exchange-data/tfrecords/{directory_name}'

        self.reset()

        self.directory_name = directory_name
        self.directory = Path(directory)

        if end_date is None:
            self.stop_date = self.now()
        else:
            self.stop_date = end_date

        if not self.directory.exists():
            self.directory.mkdir()

        self.file_path = str(self.directory) + f'/{filename}.tfrecord'

        self._last_datetime = self.start_date

    def reset(self, **kwargs):
        if self.step_count > 0:
            alog.debug('##### reset ######')
            alog.info(alog.pformat(self.summary()))

        _kwargs = self._args['kwargs']
        del self._args['kwargs']
        _kwargs = {**self._args, **_kwargs, **kwargs}

        del _kwargs['self']

        new_instance = OrderBookTradingEnv(**_kwargs)
        self.__dict__ = new_instance.__dict__

        self.get_observation()

        return self.last_observation

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

        return position

    def run(self):
        with TFRecordWriter(self.file_path, TFRecordCompressionType.GZIP) as writer:
            while self._last_datetime < self.stop_date:
                self.write_observation(writer)

    def write_observation(self, writer):
        self.get_observation()
        self.step_count += 1

        data = dict(
            datetime=self.BytesFeature(self.last_datetime),
            frame=self.floatFeature(self.frames[-2].flatten()),
            # diff=self.floatFeature([self.diff]),
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


class OrderBookTFRecordWorkers(DateRangeSplitWorkers):
    worker_class = OrderBookTFRecord

    def __init__(self, clear, directory_name, **kwargs):
        super().__init__(
            directory_name=directory_name,
            **kwargs
        )

        if clear:
            try:
                shutil.rmtree(directory)
            except Exception:
                pass


class RepeatOrderBookTFRecordWorkers(Messenger):
    def __init__(self, repeat_interval, directory_name, max_files, split, **kwargs):
        self.max_files = max_files
        self.split = split
        self.directory = Path(f'{Path.home()}/.exchange-data/tfrecords/{directory_name}')
        self.repeat_interval = repeat_interval
        kwargs['directory_name'] = directory_name
        kwargs['directory'] = self.directory
        kwargs['split'] = split
        self.kwargs = kwargs
        super(RepeatOrderBookTFRecordWorkers, self).__init__(**kwargs)
        self.on(repeat_interval, self.run_workers)

    def delete_excess_files(self):
        files = [file for file in self.directory.iterdir()]
        files.sort()

        files_to_delete = files[:(self.max_files - self.split) * -1]

        for file in files_to_delete:
            file.unlink()

    def run_workers(self, timestamp):
        start_date = DateTimeUtils.now()
        self.delete_excess_files()
        OrderBookTFRecordWorkers(**self.kwargs).run()
        self.publish('OrderBookTFRecordWorkers', str(start_date))

    def run(self):
        if self.repeat_interval:
            self.sub([self.repeat_interval])
        else:
            self.run_workers(None)


@click.command()
@click.option('--clear', '-c', is_flag=True)
@click.option('--directory-name', '-d', default='default', type=str)
@click.option('--frame-width', default=96, type=int)
@click.option('--interval', '-i', default='1h', type=str)
@click.option('--repeat-interval', '-r', default=None, type=str)
@click.option('--max-frames', '-m', default=12, type=int)
@click.option('--max-workers', '-w', default=4, type=int)
@click.option('--print-ascii-chart', '-a', is_flag=True)
@click.option('--split', '-s', default=12, type=int)
@click.option('--max-files', '-m', default=6, type=int)
@click.option('--summary-interval', '-si', default=6, type=int)
def main(**kwargs):
    record = RepeatOrderBookTFRecordWorkers(
        window_size='1m',
        is_training=False,
        **kwargs)
    record.run()


if __name__ == '__main__':
    main()
