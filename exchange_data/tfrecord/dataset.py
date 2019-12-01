#!/usr/bin/env python

import sys

import tensorflow as tf
from pathlib import Path

import click


CLASSES = [0, 1, 2]
# table = index_table_from_tensor(tf.constant(CLASSES), dtype=int64)
NUM_CLASSES = len(CLASSES)


def extract_fn(data_record):
    features = {
        'datetime': tf.io.FixedLenFeature([], tf.string),
        'frame': tf.io.FixedLenFeature([224, 224, 3], tf.float32),
        # 'diff': FixedLenFeature([1], tf.float32),
        'expected_position': tf.io.FixedLenFeature([1,], tf.int64),
    }

    sample = tf.io.parse_single_example(data_record, features)
    img = sample['frame']

    label = sample['expected_position']

    # label = tf.one_hot(label, NUM_CLASSES, dtype=tf.dtypes.int64)
    # return img, tf.argmax(label, axis=0)

    return img, label


def dataset(batch_size: int, epochs: int = 1, dataset_name='default'):
    # raise Exception()
    records_dir = f'{Path.home()}/.exchange-data/tfrecords/{dataset_name}'
    files = [str(file) for file in Path(records_dir).glob('*.tfrecord')]
    files = tf.compat.v2.data.Dataset.from_tensor_slices(files)
    # files = DatasetV2.from_tensor_slices(files)

    _dataset = files.flat_map(
        lambda filename: (tf.data.TFRecordDataset(filename, compression_type='GZIP'))
    )

    _dataset = _dataset.map(extract_fn)\
        .batch(batch_size) \
        .repeat(epochs)

    # _dataset = _dataset.map(extract_fn).window(6, drop_remainder=True)\
    #     .flat_map(lambda x, y: Dataset.zip((x.batch(6), y.skip(5).batch(1))))\
    #     .batch(2) \
    #     .repeat(epochs)

    return _dataset
    # for r in _dataset.take(10):
    #     alog.info(r)


@click.command()
# @click.option('--summary-interval', '-s', default=6, type=int)
# @click.option('--max-frames', '-m', default=12, type=int)
def main(**kwargs):
    for x in dataset(2):
        import alog
        alog.info(x)
        sys.exit(0)


if __name__ == '__main__':
    main()
