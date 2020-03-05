import json
from collections import deque

import tensorflow as tf
from tensorflow_core.core.example.feature_pb2 import Feature, Int64List, \
    FloatList, BytesList
from tensorflow_core.python.lib.io.tf_record import TFRecordWriter, \
    TFRecordCompressionType

from exchange_data.emitters.orderbook_training_data import TrainingDataBase
from exchange_data.streamers._orderbook_img import OrderbookImgStreamer
from exchange_data.tfrecord.tfrecord_directory_info import TFRecordDirectoryInfo

import alog
import re
import numpy as np


Features = tf.train.Features
Example = tf.train.Example


class OrderBookTFRecord(
    TFRecordDirectoryInfo,
    TrainingDataBase,
    OrderbookImgStreamer
):
    def __init__(
        self,
        start_date=None,
        end_date=None,
        padding=2,
        padding_after=0,
        **kwargs
    ):
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            **kwargs
        )

        filename = re.sub('[:+\s\-]', '_', str(start_date).split('.')[0])

        if end_date is None:
            self.stop_date = self.now()
        else:
            self.stop_date = end_date

        self.file_path = str(self.directory) + f'/{filename}.tfrecord'
        self.padding = padding
        self.padding_after = padding_after
        self._last_datetime = self.start_date
        self.frames = deque(maxlen=2)
        self.features = deque(maxlen=padding)
        self.done = False

    def run(self):
        with TFRecordWriter(self.file_path, TFRecordCompressionType.GZIP) \
        as writer:
            while self._last_datetime < self.stop_date:
                self.write_observation(writer)

            self.done = True

            # while len(self.features) > 0:
            #     self.write_observation(writer)

    def write_observation(self, writer):
        timestamp, best_ask, best_bid, orderbook_img = next(self)
        orderbook_img = np.asarray(json.loads(orderbook_img))

        self.last_best_ask = self.best_ask
        self.last_best_bid = self.best_bid
        self.best_ask = best_ask
        self.best_bid = best_bid
        self._last_datetime = timestamp

        self.frames.append((timestamp, best_ask, best_bid, orderbook_img))

        if len(self.frames) > 1:
            data = dict(
                datetime=self.BytesFeature(str(timestamp)),
                frame=self.floatFeature(self.frames[-2][-1].flatten()),
                best_bid=self.floatFeature([self.last_best_bid]),
                best_ask=self.floatFeature([self.last_best_ask]),
                expected_position=self.int64Feature([0]),
            )

            position = self.expected_position.value
            self.features.append((position, data))

            position_change_index = None
            position_change = None

            for i in range(len(self.features)):
                current_position = self.features[i][0]
                if current_position != 0:
                    position_change_index = i + self.padding_after
                    max_index = len(self.features) - 1
                    if position_change_index > max_index:
                        position_change_index = max_index

                    position_change = current_position

            if position_change_index is not None:
                for i in range(len(self.features)):
                    if i <= position_change_index:
                        feature = self.features[i][-1]
                        feature['expected_position'] = \
                            self.int64Feature([position_change])
                        self.features[i] = (position_change, feature)

            if len(self.features) == self.padding or self.done:
                feature = self.features[0][-1]

                example: Example = Example(
                    features=Features(feature=feature)
                )
                writer.write(example.SerializeToString())

    def int64Feature(self, value):
        return Feature(int64_list=Int64List(value=value))

    def floatFeature(self, value):
        return Feature(float_list=FloatList(value=value))

    def BytesFeature(self, value):
        return Feature(bytes_list=BytesList(value=[bytes(value, encoding='utf8')]))
