#!/usr/bin/env python

import tensorflow as tf

from pathlib import Path

import alog
import click


FixedLenFeature = tf.io.FixedLenFeature

CLASSES = [0, 1, 2]
NUM_CLASSES = len(CLASSES)


def extract_fn(data_record):
    features = dict(
        best_ask=FixedLenFeature([1], tf.float32),
        best_bid=FixedLenFeature([1], tf.float32),
        datetime=FixedLenFeature([], tf.string),
        expected_position=tf.io.FixedLenFeature([1, ], tf.int64),
        frame=FixedLenFeature([224, 224, 3], tf.float32),
    )

    return tf.io.parse_single_example(data_record, features)


def dataset(batch_size: int, epochs: int = 1, dataset_name='default', **kwargs):
    records_dir = f'{Path.home()}/.exchange-data/tfrecords/{dataset_name}'
    files = [str(file) for file in Path(records_dir).glob('*.tfrecord')]
    num_files = len(files)
    files = tf.compat.v2.data.Dataset.from_tensor_slices(files)

    _dataset = files.flat_map(
        lambda filename: (tf.data.TFRecordDataset(filename, compression_type='GZIP'))
    ).shuffle(buffer_size=num_files)

    _dataset = _dataset.map(extract_fn)\
        .batch(batch_size) \
        .prefetch(10)\
        .repeat(epochs)

    return _dataset


@click.command()
def main(**kwargs):
    for x in dataset(1, **kwargs):
        alog.info(x)


if __name__ == '__main__':
    main()
