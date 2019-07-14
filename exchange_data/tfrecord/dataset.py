from pathlib import Path
from tensorflow.io import FixedLenFeature, VarLenFeature, parse_single_example
from tensorflow.python.data import Dataset, TFRecordDataset
from tgym.envs.orderbook.ascii_image import AsciiImage

import alog
import click
import tensorflow as tf


def extract_fn(data_record):
    features = {
        'datetime': FixedLenFeature([], tf.string),
        'frame': FixedLenFeature([96, 192, 3], tf.float32),
        'expected_position': FixedLenFeature([], tf.int64),
    }

    sample = parse_single_example(data_record, features)
    img = sample['frame']
    label = sample['expected_position']
    return img, label


def dataset(epochs: int):
    records_dir = f'{Path.home()}/.exchange-data/tfrecords/'
    files = [str(file) for file in Path(records_dir).glob('*.tfrecord')]
    files = Dataset.from_tensor_slices(files)
    dataset = files.flat_map(
        lambda filename: (TFRecordDataset(filename, compression_type='GZIP'))
    )

    dataset = dataset.map(extract_fn).batch(4).repeat(epochs)

    return dataset
    # count = 0
    # for r in dataset:
    #     # alog.info(r)
    #     # alog.info(r['da
    #     alog.info(r['expected_position'])
    #     alog.info(AsciiImage(r['frame'].numpy()))
    #
    #     count += 1
    #     if count > 1:
    #         break


@click.command()
# @click.option('--summary-interval', '-s', default=6, type=int)
# @click.option('--max-frames', '-m', default=12, type=int)
def main(**kwargs):
    dataset()


if __name__ == '__main__':
    main()
