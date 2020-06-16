#!/usr/bin/env python
import re

import tensorflow as tf

from pathlib import Path
import tensorflow_datasets as tfds
import alog
import click
import numpy as np

from exchange_data.utils import DateTimeUtils

FixedLenFeature = tf.io.FixedLenFeature

CLASSES = [0, 1, 2]
NUM_CLASSES = len(CLASSES)


def extract_fn(data_record):
    alog.info(data_record)

    features = dict(
        best_ask=FixedLenFeature([1], tf.float32),
        best_bid=FixedLenFeature([1], tf.float32),
        datetime=FixedLenFeature([], tf.string),
        frame=FixedLenFeature([80, 1], tf.float32),
    )

    data = tf.io.parse_single_example(data_record, features)

    return data


def dataset(batch_size: int, skip=0, epochs: int = 1, dataset_name='default',
            **kwargs):

    records_dir = f'{Path.home()}/.exchange-data/tfrecords/{dataset_name}'
    files = [str(file) for file in Path(records_dir).glob('*.tfrecord')]

    file_dates = sorted([
        (DateTimeUtils.parse_db_timestamp(int(re.findall(f'\d+', file)[0])), file)
        for file in files])

    file_dates = reversed(file_dates)

    files = [fd[-1] for fd in file_dates]

    # alog.info(alog.pformat(files))

    files = tf.compat.v2.data.Dataset.from_tensor_slices(files)

    def read_file(filename):
        return tf.data.TFRecordDataset(
            filename, compression_type='GZIP')

    _dataset = files.flat_map(lambda filename: read_file(filename))

    _dataset = _dataset.map(map_func=extract_fn) \
        .batch(batch_size) \
        .skip(skip) \
        .prefetch(10) \
        .repeat(epochs)

    return _dataset


@click.command()
def main(**kwargs):
    count = 0
    max = 0

    for data in tfds.as_numpy(dataset(1, **kwargs)):
        alog.info(data['datetime'])
        frame_max = np.amax(data['frame'])
        if frame_max > max:
            max = frame_max

        count += 1

    alog.info(max)


if __name__ == '__main__':
    main()
