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

from exchange_data.trading import Positions

Features = tf.train.Features
Example = tf.train.Example


class OrderBookTFRecord(
    TFRecordDirectoryInfo,
    TrainingDataBase,
    OrderbookImgStreamer
):
    def __init__(
        self,
        side,
        start_date=None,
        end_date=None,
        padding=2,
        padding_after=0,
        **kwargs
    ):
        super().__init__(
            side=side,
            start_date=start_date,
            end_date=end_date,
            **kwargs
        )

        filename = re.sub('[:+\s\-]', '_', str(start_date).split('.')[0])

        if end_date is None:
            self.stop_date = self.now()
        else:
            self.stop_date = end_date

        self.side = side
        self.file_path = str(self.directory) + f'/{filename}.tfrecord'
        padding = padding
        self.padding_window = padding + padding_after
        self.padding = padding
        self.padding_after = padding_after
        self._last_datetime = self.start_date
        self.frames = deque(maxlen=2)
        self.features = []
        self.done = False

    def run(self):
        with TFRecordWriter(self.file_path, TFRecordCompressionType.GZIP) \
        as writer:
            while self._last_datetime < self.stop_date:
                self.queue_obs()

            # some data transformations here
            self.window_position_change()

            for d in self.features:
                self.write_observation(writer, d)

    def window_position_change(self):
        change_indexes = []

        for i in range(len(self.features)):
            feature = self.features[i]
            current_position = feature[0]

            if current_position != 0:
                position = feature[-1]['expected_position']
                change_indexes.append((i, position))

        max_index = len(self.features) - 1

        self.features = [feature[-1] for feature in self.features]

        for position_change in change_indexes:
            i, position = position_change
            left_padding_index = i - self.padding
            right_padding_index = i + self.padding_after

            if left_padding_index < 0:
                left_padding_index = 0

            if right_padding_index > max_index:
                right_padding_index = max_index

            for ix in range(left_padding_index, right_padding_index + 1):
                feature = self.features[ix]
                feature['expected_position'] = position
                self.features[ix] = feature

    def queue_obs(self):
        timestamp, best_ask, best_bid, orderbook_img = next(self)
        orderbook_img = np.asarray(json.loads(orderbook_img))

        self.last_best_ask = self.best_ask
        self.last_best_bid = self.best_bid
        self.best_ask = best_ask
        self.best_bid = best_bid
        self._last_datetime = timestamp

        self.frames.append((timestamp, best_ask, best_bid, orderbook_img))

        if len(self.frames) > 1:
            position = self.expected_position.value

            if position != Positions[self.side].value and \
                position != Positions.Flat.value:
                position = Positions.Flat.value

            data = dict(
                datetime=self.BytesFeature(str(timestamp)),
                frame=self.floatFeature(self.frames[-2][-1].flatten()),
                best_bid=self.floatFeature([self.last_best_bid]),
                best_ask=self.floatFeature([self.last_best_ask]),
                expected_position=self.int64Feature([position]),
            )
            self.features.append((position, data))

    def write_observation(self, writer, features):
        # alog.info(features['datetime'])
        # alog.info(features['expected_position'])

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
