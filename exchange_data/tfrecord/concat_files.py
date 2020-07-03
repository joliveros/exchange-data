#!/usr/bin/env python
import os
import shutil

from tensorflow.core.example.example_pb2 import Example
from tensorflow.core.example.feature_pb2 import Features, Feature, FloatList, \
    Int64List, BytesList
from tensorflow.python.lib.io.tf_record import TFRecordWriter, \
    TFRecordCompressionType

from exchange_data.emitters.backtest import BackTest
from exchange_data.tfrecord.dataset import dataset
import alog
import click
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
from pathlib import Path


def write_observation(writer, features):
    example: Example = Example(
        features=Features(feature=features)
    )
    writer.write(example.SerializeToString())


def int64Feature(value):
    return Feature(int64_list=Int64List(value=value))


def floatFeature(value):
    return Feature(float_list=FloatList(value=value))


def BytesFeature(value):
    return Feature(bytes_list=BytesList(value=[bytes(value, encoding='utf8')]))

def filename(dataset_name, suffix):
    dir = f'{Path.home()}/.exchange-data/tfrecords/{dataset_name}'
    Path(dir).mkdir(exist_ok=True)
    file = f'{dir}/data_{suffix}.tfrecord'
    temp_file = f'{file}.temp'

    try:
        os.remove(file)

    except:
        pass
    try:
        os.remove(temp_file)

    except:
        pass

    return file,  temp_file


def write(dataset_name, suffix, df):
    file, temp_file = filename(dataset_name, suffix)

    with TFRecordWriter(temp_file, TFRecordCompressionType.GZIP) \
        as writer:

        for name, row in df.iterrows():
            data = dict(
                expected_position=int64Feature([int(row['expected_position'])]),
                frame=floatFeature(row['frame_sequence'].flatten()),
            )

            write_observation(writer, data)

    shutil.move(temp_file, file)


def convert(
    window_size,
    group_by,
    symbol=None,
    expected_position_length=4,
    sequence_length=48,
    labeled_ratio=0.5,
    dataset_name='default',
    min_change=2.0,
    **kwargs
):
    backtest = BackTest(**{
        'database_name': 'binance',
        'depth': 40,
        'sequence_length': 48,
        'symbol': symbol,
        'volume_max': 10000.0,
        'group_by': group_by,
        'window_size': window_size
    }, **kwargs)

    df = backtest.df
    df.rename(columns={'orderbook_img': 'frame'}, inplace=True)

    alog.info(df)

    df['avg_price'] = (df['best_ask'] + df['best_bid']) / 2

    df = df.drop(columns=['best_bid', 'best_ask'])
    df.dropna(how='any', inplace=True)
    df = df.set_index('time')
    df = df.sort_index()
    df = df.reset_index(drop=False)

    min_price = df['avg_price'].min()

    min_change = min_price * (min_change / 100)

    df['expected_position'] = np.where(
        (df['avg_price']) - df['avg_price'].shift(1) > min_change, 1, 0)

    for i in range(0, len(df)):

        expected_position = df.loc[i, 'expected_position'] == 1

        if expected_position:
            for x in range(0, expected_position_length - 1):
                y = i - x
                df.loc[y, 'expected_position'] = 1

    alog.info(df)

    def frame_sequence(row):
        frames = []
        i = 0

        while len(frames) < sequence_length:

            current_row = df.iloc[row.name - i]
            frame = current_row['frame']
            datetime = current_row['time'].to_datetime64()

            if type(frame) is np.ndarray:
                frames = [frame] + frames

            i += 1

        frames = np.asarray(frames)

        row['frame_sequence'] = frames

        return row

    df = df.apply(frame_sequence, axis=1)

    df.dropna(how='any', inplace=True)
    df.drop(columns=['frame'])

    labeled_df = df[df['expected_position'] == 1]

    labeled_count = labeled_df.shape[0]
    unlabeled_df = df[df['expected_position'] == 0]

    unlabeled_count = int(labeled_count * (1 / labeled_ratio) * (1 - labeled_ratio))

    fraction = unlabeled_count / unlabeled_df.shape[0]

    if fraction > 1.0:
        fraction = 1.0
    unlabeled_df = unlabeled_df.sample(frac=fraction)

    write(dataset_name, 'labeled', labeled_df)
    write(dataset_name, 'unlabeled', unlabeled_df)

    return labeled_count


@click.command()
@click.argument('symbol', nargs=1, required=True)
@click.option('--dataset-name', '-d', default='default', type=str)
@click.option('--interval', '-i', default='1d', type=str)
@click.option('--window-size', '-w', default='2h', type=str)
@click.option('--group-by', '-g', default='1m', type=str)
@click.option('--expected-position-length', '-e', default=4, type=int)
@click.option('--labeled-ratio', '-l', default=0.5, type=float)
@click.option('--min-change', '-m', default=2.0, type=float)
@click.option('--sequence-length', '-s', default=48, type=int)
@click.option('--plot', '-p', is_flag=True)
def main(**kwargs):
    convert(**kwargs)


if __name__ == '__main__':
    main()
