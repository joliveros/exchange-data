#!/usr/bin/env python
import time

import alog

from exchange_data.tfrecord.date_range_split_workers import DateRangeSplitWorkers
from exchange_data.tfrecord.orderbook_tf_record import OrderBookTFRecord
from pytimeparse.timeparse import timeparse

import click
import shutil

from exchange_data.tfrecord.tfrecord_directory_info import TFRecordDirectoryInfo
from exchange_data.trading import Positions


class OrderBookTFRecordWorkers(TFRecordDirectoryInfo, DateRangeSplitWorkers):
    worker_class = OrderBookTFRecord

    def __init__(self, depth, clear, **kwargs):
        channel_name=f'orderbook_img_frame_XBTUSD_{depth}'
        super().__init__(depth=depth, channel_name=channel_name, **kwargs)

        if clear:
            try:
                shutil.rmtree(self.directory)
            except Exception:
                pass


@click.command()
@click.option('--clear', '-c', is_flag=True)
@click.option('--overwrite', '-o', is_flag=True)
@click.option('--directory-name', '-d', default='default', type=str)
@click.option('--interval', '-i', default='1h', type=str)
@click.option('--limit', '-l', default=0, type=int)
@click.option('--depth', default=21, type=int)
@click.option('--padding', default=2, type=int)
@click.option('--padding-after', default=0, type=int)
@click.option('--max-workers', '-w', default=4, type=int)
@click.option('--print-ascii-chart', '-a', is_flag=True)
@click.option('--summary-interval', '-si', default=6, type=int)
@click.option('--position-ratio', default=1.0, type=float)
@click.option('--window-size', '-g', default='1m', type=str)
@click.option('--record-window', '-r', default='15s', type=str)
@click.option('--side', type=click.Choice(Positions.__members__),
                default='Short')
def main(**kwargs):
    record = OrderBookTFRecordWorkers(
        database_name='bitmex',
        is_training=False,
        **kwargs)

    record.run()


if __name__ == '__main__':
    main()


