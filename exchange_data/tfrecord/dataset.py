from pathlib import Path

from tensorflow import one_hot
from tensorflow.io import FixedLenFeature, VarLenFeature, parse_single_example
from tensorflow.python.data import Dataset, TFRecordDataset
from tensorflow.python.framework import dtypes
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.ops.lookup_ops import index_table_from_tensor

from tgym.envs.orderbook.ascii_image import AsciiImage

import alog
import click
import tensorflow as tf

CLASSES = [0, 1, 2]
table = index_table_from_tensor(tf.constant(CLASSES), dtype=dtypes.int64)
NUM_CLASSES = len(CLASSES)


def extract_fn(data_record):
    features = {
        'datetime': FixedLenFeature([], tf.string),
        'frame': FixedLenFeature([96, 192, 3], tf.float32),
        'expected_position': FixedLenFeature([], tf.int64),
    }

    sample = parse_single_example(data_record, features)
    img = sample['frame']

    label = sample['expected_position']

    label = one_hot(label, NUM_CLASSES, dtype=dtypes.int64)
    return img, label


def dataset(epochs: int = 1):
    records_dir = f'{Path.home()}/.exchange-data/tfrecords/'
    files = [str(file) for file in Path(records_dir).glob('*.tfrecord')]
    files = Dataset.from_tensor_slices(files)
    _dataset = files.flat_map(
        lambda filename: (TFRecordDataset(filename, compression_type='GZIP'))
    )

    _dataset = _dataset.map(extract_fn).window(6, drop_remainder=True)\
        .flat_map(lambda x, y: Dataset.zip((x.batch(6), y.skip(5).batch(1))))\
        .repeat(epochs)

    return _dataset
    # for r in _dataset.take(1000):
    #     alog.info(r)


@click.command()
# @click.option('--summary-interval', '-s', default=6, type=int)
# @click.option('--max-frames', '-m', default=12, type=int)
def main(**kwargs):
    dataset()


if __name__ == '__main__':
    main()
