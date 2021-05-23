#!/usr/bin/env python

import json
import shutil
import time
from collections import deque
from pathlib import Path

import click
import tensorflow as tf
from tensorflow.core.example.feature_pb2 import Feature, Int64List, FloatList, \
    BytesList
from tensorflow.python.lib.io.tf_record import TFRecordWriter, \
    TFRecordCompressionType

from exchange_data.data.orderbook_frame import OrderBookFrame
from exchange_data.emitters.orderbook_training_data import TrainingDataBase
from exchange_data.streamers._orderbook_level import OrderBookLevelStreamer
from exchange_data.tfrecord.tfrecord_directory_info import TFRecordDirectoryInfo

import alog
import re
import numpy as np

from exchange_data.trading import Positions

Features = tf.train.Features
Example = tf.train.Example


class NoOrderLevelsException(Exception):
    pass


class OrderBookTFRecordBase(TFRecordDirectoryInfo, TrainingDataBase):
    def __init__(
        self,
        side,
        overwrite: bool,
        **kwargs
    ):
        super().__init__(**kwargs)

        filename = str(int(time.time() * 1000))

        self.side = side
        self.file_path = str(self.directory) + f'/{filename}.tfrecord'
        self.temp_file_path = str(self.directory) + f'/{filename}.temp'

        if not overwrite and Path(self.file_path).exists():
            raise Exception('File Exists')

        self._last_datetime = self.start_date
        self.frames = deque(maxlen=2)
        self.features = []
        self.done = False

    def write_observation(self, writer, features):
        example: Example = Example(
            features=Features(feature=features)
        )

        writer.write(example.SerializeToString())

    def int64Feature(self, value):
        return Feature(int64_list=Int64List(value=value))

    def floatFeature(self, value):
        return Feature(float_list=FloatList(value=value))

    def BytesFeature(self, value):
        return Feature(bytes_list=BytesList(value=[bytes(value, encoding='utf8')]))


class OrderBookTFRecord(OrderBookFrame, OrderBookTFRecordBase):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(
            **kwargs
        )
        OrderBookTFRecordBase.__init__(self, **kwargs)

    def run(self):
        with TFRecordWriter(self.temp_file_path, TFRecordCompressionType.GZIP) \
        as writer:
            frame = self.frame
            for i in range(0, len(frame)):
                row = frame.iloc[i]

                data = dict(
                    datetime=self.BytesFeature(str(timestamp)),
                    frame=self.floatFeature(self.frames[-1][-1].flatten()),
                    best_bid=self.floatFeature([row.best_bid]),
                    best_ask=self.floatFeature([row.best_ask]),
                )
                self.features.append(data)

                self.write_observation(writer, d)

        shutil.move(self.temp_file_path, self.file_path)


@click.command()
@click.option('--database-name', default='binance', type=str)
@click.option('--depth', default=48, type=int)
@click.option('--directory-name', '-d', default='default', type=str)
@click.option('--group-by', '-g', default='30s', type=str)
@click.option('--interval', '-i', default='10m', type=str)
@click.option('--overwrite', '-o', is_flag=True)
@click.option('--print-ascii-chart', '-a', is_flag=True)
@click.option('--record-window', '-r', default='15s', type=str)
@click.option('--sequence-length', '-l', default=24, type=int)
@click.option('--summary-interval', '-si', default=6, type=int)
@click.option('--symbol', '-s', default='', type=str)
@click.option('--window-size', '-g', default='1m', type=str)
@click.option('--side', type=click.Choice(Positions.__members__),
                default='Short')
def main(**kwargs):
    record = OrderBookTFRecord(**kwargs)

    record.run()

if __name__ == '__main__':
    main()