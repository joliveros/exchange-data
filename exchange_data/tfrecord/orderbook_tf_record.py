import json
import shutil
import time
from collections import deque
from pathlib import Path

import tensorflow as tf
# from tensorflow_core.core.example.feature_pb2 import Feature, Int64List, \
#     FloatList, BytesList
# from tensorflow_core.python.lib.io.tf_record import TFRecordWriter, \
#     TFRecordCompressionType
from tensorflow.core.example.feature_pb2 import Feature, Int64List, FloatList, \
    BytesList
from tensorflow.python.lib.io.tf_record import TFRecordWriter, \
    TFRecordCompressionType
from exchange_data.emitters.orderbook_training_data import TrainingDataBase
from exchange_data.streamers._orderbook_level import OrderBookLevelStreamer
from exchange_data.tfrecord.tfrecord_directory_info import TFRecordDirectoryInfo

import alog
import re
import numpy as np

from exchange_data.trading import Positions

Features = tf.train.Features
Example = tf.train.Example


class OrderBookTFRecordBase(TFRecordDirectoryInfo, TrainingDataBase):
    def __init__(
        self,
        side,
        overwrite: bool,
        start_date=None,
        end_date=None,
        padding=2,
        padding_after=0,
        **kwargs
    ):
        super().__init__(**kwargs)

        # filename = re.sub('[:+\s\-]', '_', str(start_date).split('.')[0])
        filename = str(int(time.time() * 1000))

        if end_date is None:
            self.stop_date = self.now()
        else:
            self.stop_date = end_date

        self.side = side
        self.file_path = str(self.directory) + f'/{filename}.tfrecord'
        self.temp_file_path = str(self.directory) + f'/{filename}.temp'

        if not overwrite and Path(self.file_path).exists():
            raise Exception('File Exists')

        padding = padding
        self.padding_window = padding + padding_after
        self.padding = padding
        self.padding_after = padding_after
        self._last_datetime = self.start_date
        self.frames = deque(maxlen=2)
        self.features = []
        self.done = False

    def queue_obs(self):
        timestamp, best_ask, best_bid, orderbook_levels = next(self)
        orderbook_levels = np.asarray(json.loads(orderbook_levels))
        orderbook_levels = np.delete(orderbook_levels, 0, 1)
        orderbook_levels[0][0] = np.flip(orderbook_levels[0][0])
        orderbook_levels[1] =  orderbook_levels[1] * -1
        max = 3.0e6
        orderbook_levels = np.reshape(orderbook_levels, (80, 1)) / max
        orderbook_levels = np.clip(orderbook_levels, a_min=0.0, a_max=max)

        self.best_ask = best_ask
        self.best_bid = best_bid

        self._last_datetime = timestamp

        self.frames.appendleft((timestamp, best_ask, best_bid,
                                orderbook_levels))

        data = dict(
            datetime=self.BytesFeature(str(timestamp)),
            frame=self.floatFeature(self.frames[-1][-1].flatten()),
            best_bid=self.floatFeature([self.best_bid]),
            best_ask=self.floatFeature([self.best_ask]),
        )
        self.features.append(data)

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


class OrderBookTFRecord(OrderBookLevelStreamer, OrderBookTFRecordBase):
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
            while self._last_datetime < self.stop_date:
                self.queue_obs()

            for d in self.features:
                self.write_observation(writer, d)

        shutil.move(self.temp_file_path, self.file_path)
