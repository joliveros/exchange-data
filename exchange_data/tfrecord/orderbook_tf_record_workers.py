#!/usr/bin/env python
import time

import alog

from exchange_data.tfrecord.date_range_split_workers import DateRangeSplitWorkers
from exchange_data.tfrecord.orderbook_tf_record import OrderBookTFRecord
from pytimeparse.timeparse import timeparse

import click
import shutil

from exchange_data.tfrecord.tfrecord_directory_info import TFRecordDirectoryInfo


class OrderBookTFRecordWorkers(TFRecordDirectoryInfo, DateRangeSplitWorkers):
    worker_class = OrderBookTFRecord

    def __init__(self, clear, **kwargs):
        super().__init__(**kwargs)

        if clear:
            try:
                shutil.rmtree(self.directory)
            except Exception:
                pass


@click.command()
@click.option('--clear', '-c', is_flag=True)
@click.option('--directory-name', '-d', default='default', type=str)
@click.option('--interval', '-i', default='1h', type=str)
@click.option('--limit', '-l', default=0, type=int)
@click.option('--padding', default=2, type=int)
@click.option('--padding-after', default=0, type=int)
@click.option('--max-workers', '-w', default=4, type=int)
@click.option('--print-ascii-chart', '-a', is_flag=True)
@click.option('--summary-interval', '-si', default=6, type=int)
@click.option('--window-size', '-g', default='1m', type=str)
@click.option('--record-window', '-r', default='15s', type=str)
def main(**kwargs):
    record = OrderBookTFRecordWorkers(
        database_name='bitmex',
        is_training=False,
        channel_name='orderbook_img_frame_XBTUSD',
        **kwargs)

    record.run()
    time.sleep(timeparse('30s'))


if __name__ == '__main__':
    main()


