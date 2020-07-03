#!/usr/bin/env python

from exchange_data.utils import DateTimeUtils
from pathlib import Path
import alog
import click
import re
import tensorflow as tf
import tensorflow_datasets as tfds

FixedLenFeature = tf.io.FixedLenFeature

CLASSES = [0, 1, 2]
NUM_CLASSES = len(CLASSES)


def read_file(filename):
    return tf.data.TFRecordDataset(
        filename, compression_type='GZIP')

def dataset(batch_size: int, sequence_length=48, skip=0, epochs: int = 1,
            dataset_name='default', filename='data',
            **kwargs):
    def extract_fn(data_record):
        # alog.info(data_record)

        features = dict(
            expected_position=FixedLenFeature([], tf.int64),
            frame=FixedLenFeature([sequence_length, 80, 1], tf.float32),
        )

        data = tf.io.parse_single_example(data_record, features)

        return data['frame'], data['expected_position']

    data_file = f'{Path.home()}/.exchange-data/tfrecords/' \
                  f'{dataset_name}/{filename}.tfrecord'

    alog.info(data_file)

    _dataset = read_file(data_file)

    _dataset = _dataset.map(map_func=extract_fn) \
        .batch(batch_size) \
        .skip(skip) \
        .repeat(epochs)

    return _dataset


@click.command()
@click.option('--dataset-name', '-d', type=str, default='default')
@click.option('--filename', '-f', type=str, default='default')
def main(**kwargs):
    count = 0
    max = 0

    for frame, expected_position in tfds.as_numpy(dataset(batch_size=2,
                                                         epochs=1, **kwargs)):
        alog.info(frame)
        alog.info(frame.shape)
        # alog.info(expected_position)
        # raise Exception()
        count += 1

    alog.info(max)


if __name__ == '__main__':
    main()
