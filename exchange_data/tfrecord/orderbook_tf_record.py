import shutil
from datetime import timedelta

from exchange_data.streamers._bitmex import OutOfFramesException
from exchange_data.utils import DateTimeUtils
from multiprocessing import Process
from pathlib import Path
from pytimeparse.timeparse import timeparse
from tensorflow.core.example.example_pb2 import Example
from tensorflow.core.example.feature_pb2 import Features, FloatList, BytesList, Feature, Int64List
from tensorflow.python.lib.io.tf_record import TFRecordWriter, TFRecordCompressionType
from tgym.envs import OrderBookTradingEnv
from tgym.envs.orderbook.utils import Positions
from time import sleep

import alog
import click
import re
import sys


class OrderBookTFRecord(OrderBookTradingEnv):
    def __init__(self, directory_name, filename, start_date=None, end_date=None, **kwargs):
        OrderBookTradingEnv.__init__(
            self,
            random_start_date=False,
            use_volatile_ranges=False,
            start_date=start_date,
            **kwargs
        )

        self.reset()

        if end_date is None:
            self.stop_date = self.now()
        else:
            self.stop_date = end_date
        self.directory = Path(f'{Path.home()}/.exchange-data/tfrecords/{directory_name}/')

        if not self.directory.exists():
            self.directory.mkdir()

        self.file_path = str(self.directory) + f'/{filename}.tfrecord'

    def reset(self, **kwargs):
        alog.info(self.start_date)
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
        # alog.info(self.frames[-2].shape)
        # raise Exception()

        data = dict(
            datetime=self.BytesFeature(self.last_datetime),
            frame=self.floatFeature(self.frames[-2].flatten()),
            diff=self.floatFeature([self.diff]),
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
@click.option('--split', '-s', default=12, type=int)
@click.option('--max-frames', '-m', default=12, type=int)
@click.option('--max-workers', '-w', default=4, type=int)
@click.option('--print-ascii-chart', '-a', is_flag=True)
@click.option('--frame-width', default=96, type=int)
@click.option('--clear', '-c', is_flag=True)
@click.option('--name', '-n', default='default', type=str)
def main(interval, split, max_workers, clear, name, **kwargs):
    now = DateTimeUtils.now()
    start_date = now - timedelta(seconds=timeparse(interval))
    dates = DateTimeUtils.split_range_into_datetimes(start_date, now, split)
    intervals = []
    directory = f'{Path.home()}/.exchange-data/tfrecords/{name}'

    if clear:
        try:
            shutil.rmtree(directory)
        except Exception:
            pass

    for i in range(len(dates)):
        if i < len(dates) - 1:
            intervals += [(dates[i], dates[i + 1] - timedelta(seconds=1))]

    def replay(start_date, end_date, **kwargs):
        filename = re.sub('[:+\s\-]', '_', str(start_date).split('.')[0])
        record = OrderBookTFRecord(
            window_size='1m',
            is_training=False,
            start_date=start_date,
            end_date=end_date,
            filename=filename,
            directory_name=name,
            **kwargs
        )

        record.run()

    workers = []

    while True:
        if len(workers) < max_workers and len(intervals) > 0:
            args = intervals.pop()
            alog.debug(f'#### ranges left {len(intervals)} ####')
            worker = Process(target=replay, args=args, kwargs=kwargs)
            worker.start()
            alog.debug(worker)
            workers.append(worker)
        if len(intervals) == 0 and len(workers) == 0:
            sys.exit(0)

        workers = [w for w in workers if w.is_alive()]

        sleep(1)

if __name__ == '__main__':
    main()

