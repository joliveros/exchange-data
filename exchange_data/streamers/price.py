#!/usr/bin/env python

from datetime import timedelta
from exchange_data.streamers._bitmex import BitmexStreamer
from exchange_data.utils import DateTimeUtils

import alog
import click
from pytimeparse.timeparse import timeparse


class PriceStreamer(BitmexStreamer):
    def __init__(
        self,
        symbol,
        interval,
        depth=40,
        group_by='2s',
        **kwargs
    ):
        self.end_date = DateTimeUtils.now()
        self.start_date = self.end_date - timedelta(seconds=timeparse(interval))

        super().__init__(
            start_date = self.start_date,
            end_date = self.end_date,
            **kwargs)

        self.groupby = group_by
        self.last_timestamp = self.start_date
        self.channel_name = f'{symbol}_OrderBookFrame_depth_{depth}'
        self.current_query = self.get_orderbook_frames()

    def orderbook_frame_query(self):
        start_date = self.start_date
        end_date = self.end_date

        start_date = self.format_date_query(start_date)
        end_date = self.format_date_query(end_date)

        query = f'SELECT first("best_ask") AS best_ask, first("best_bid") AS ' \
                f'best_bid FROM {self.channel_name} ' \
            f'WHERE time > {start_date} AND time <= {end_date} GROUP BY ' \
                f'time({self.groupby});'

        alog.info(query)

        return self.query(query)

    def get_orderbook_frames(self):
        orderbook = self.orderbook_frame_query()
        for data in orderbook.get_points(self.channel_name):
            timestamp = DateTimeUtils.parse_db_timestamp(data['time'])

            if self.last_timestamp != timestamp:
                best_bid = data['best_bid']
                best_ask = data['best_ask']
                self.last_timestamp = timestamp
                yield timestamp, best_ask, best_bid

            self.last_timestamp = timestamp

    def send(self, *args):
        if self.last_timestamp < self.stop_date:
            try:
                return next(self.current_query)
            except StopIteration as e:
                self._set_next_window()
                if self.end_date > self.stop_date:
                    raise StopIteration()
                self.current_query = self.get_orderbook_frames()
                return None, None, None

        raise StopIteration()


@click.command()
@click.option('--window-size',
              '-w',
              type=str,
              default='1h',
              help='Window size i.e. "1m"')
@click.option('--database-name', '-d', default='binance', type=str)
@click.option('--group-by', '-g', default='1m', type=str)
@click.option('--interval', '-i', default='2h', type=str)
@click.argument('symbol', type=str)
def main(**kwargs):
    streamer = PriceStreamer(**kwargs)

    for timestamp, best_ask, best_bid in streamer:
        alog.info((str(timestamp), best_ask, best_bid))


if __name__ == '__main__':
    main()
