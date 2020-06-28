#!/usr/bin/env python
import os
import shutil

from tensorflow.core.example.example_pb2 import Example
from tensorflow.core.example.feature_pb2 import Features, Feature, FloatList, \
    Int64List, BytesList
from tensorflow.python.lib.io.tf_record import TFRecordWriter, \
    TFRecordCompressionType

from exchange_data.tfrecord.dataset import dataset
import alog
import click
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
from pathlib import Path


def convert(
    expected_position_length=4,
    sequence_length=48,
    labeled_ratio=0.5,
    dataset_name='default',
    min_change=2.0
):
    file = f'{Path.home()}/.exchange-data/tfrecords/' \
                  f'{dataset_name}/data.tfrecord'
    try:
        os.remove(file)
    except:
        pass

    frames = []


    for data in tfds.as_numpy(dataset(batch_size=1, epochs=1,
                                  dataset_name=dataset_name)):
        frames += [data]

        # if len(frames) <= 500:
        #     frames += [data]
        # else:
        #     break

    df = pd.DataFrame(frames)
    df['best_ask'] = df['best_ask'].apply(lambda x: x[0][0])
    df['best_bid'] = df['best_bid'].apply(lambda x: x[0][0])
    df['datetime'] = df['datetime'].apply(lambda x: x[0].decode())
    df['datetime'] = pd.to_datetime(df['datetime'])

    df['avg_price'] = (df['best_ask'] + df['best_bid']) / 2

    df = df.drop(columns=['best_bid', 'best_ask'])
    df.dropna(how='any', inplace=True)
    df = df.set_index('datetime')
    df = df.sort_index()
    df = df.reset_index(drop=False)

    df['frame'] = df['frame'].apply(lambda f: f[0])

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
            datetime = current_row['datetime'].to_datetime64()

            if type(frame) is np.ndarray:
                frames = [frame] + frames

            i += 1


        frames = np.asarray(frames)

        row['frame_sequence'] = frames

        return row

    df = df.apply(frame_sequence, axis=1)

    df.dropna(how='any', inplace=True)
    df.drop(columns=['frame'])

    # alog.info(df)
    # alog.info(df.shape)

    labeled_df = df[df['expected_position'] == 1]

    labeled_count = labeled_df.shape[0]
    unlabeled_df = df[df['expected_position'] == 0]

    unlabeled_count = labeled_count * (1 / labeled_ratio) * (1 - labeled_ratio)

    alog.info((labeled_ratio, labeled_count, unlabeled_count))

    unlabeled_df = unlabeled_df.sample(frac=unlabeled_count / unlabeled_df.shape[0])

    df = pd.concat([labeled_df, unlabeled_df]).reset_index(drop=True)

    alog.info(df)

    temp_file = f'{file}.temp'

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

    with TFRecordWriter(temp_file, TFRecordCompressionType.GZIP) \
        as writer:

        for name, row in df.iterrows():
            data = dict(
                expected_position=int64Feature([int(row['expected_position'])]),
                frame=floatFeature(row['frame_sequence'].flatten()),
            )

            write_observation(writer, data)

    shutil.move(temp_file, file)

    # epoch_count = 0

    # while epoch_count <= epochs:
    #
    #     shuffled_df = pd.concat([labeled_df, unlabeled_df]).sample(
    #         frac=1.0).reset_index(drop=True)
    #
    #     for i in range(0, len(shuffled_df)):
    #         row = shuffled_df.loc[i]
    #         yield row['frame'], row['expected_position']
    #
    #     epoch_count += 1



@click.command()
@click.option('--dataset-name', '-d', default='default', type=str)
@click.option('--expected-position-length', '-e', default=4, type=int)
@click.option('--labeled-ratio', '-l', default=0.5, type=float)
@click.option('--min-change', '-m', default=2.0, type=float)
@click.option('--sequence-length', '-s', default=48, type=int)
def main(**kwargs):
    convert(**kwargs)


if __name__ == '__main__':
    main()
