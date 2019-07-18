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

    _dataset = _dataset.map(extract_fn).window(6).repeat(epochs)

    # alog.info(dataset.output_shapes)

    # return dataset
    count = 0
    for r in _dataset.take(2):
        alog.info(r)
        # # alog.info(r['da
        # alog.info(r['expected_position'])
        # alog.info(AsciiImage(r['frame'].numpy()))

        count += 1
        if count > 1:
            break


@click.command()
# @click.option('--summary-interval', '-s', default=6, type=int)
# @click.option('--max-frames', '-m', default=12, type=int)
def main(**kwargs):
    dataset()


if __name__ == '__main__':
    main()
