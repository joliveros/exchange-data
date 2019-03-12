import json
import sys
from copy import copy

import click
from cached_property import cached_property
from datetime import datetime, timedelta
from dateutil.tz import tz
from exchange_data.channels import BitmexChannels
from exchange_data.emitters.bitmex import BitmexOrderBookEmitter
from exchange_data.streamers._bitmex import BitmexOrderBookChannels
from exchange_data.utils import DateTimeUtils

import alog


class OrderBookPlayBack(BitmexOrderBookEmitter, DateTimeUtils):
    def __init__(
        self,
        start_date=None,
        end_date=None,
        max_cache_len: int = 60 * 45,
        query_interval: int = 60 * 24,
        **kwargs
    ):
        super().__init__(
            symbol=BitmexChannels.XBTUSD,
            subscriptions_enabled=False,
            **kwargs
        )
        self.query_interval = timedelta(minutes=query_interval)
        self.max_cache_len = max_cache_len
        self._measurements = []
        self.measurement = 'data'

        if start_date is None:
            self.start_date = self.min_date
        else:
            self.start_date = start_date

        self.start_date = self.start_date.replace(microsecond=0)
        self._next_tick = self.start_date

        if end_date:
            self.end_date = end_date
        else:
            self.end_date = self.now()

    @cached_property
    def min_date(self):
        start_date = datetime.fromtimestamp(0, tz=tz.tzutc())

        result = self.oldest_frame_query(start_date, self.now())

        for item in result.get_points(self.measurement):
            timestamp = self.parse_db_timestamp(item['time'])
            return timestamp

        raise Exception('Database has no data.')

    def oldest_frame_query(self, start_date, end_date):
        start_date = self.format_date_query(start_date)
        end_date = self.format_date_query(end_date)

        query = f'SELECT * FROM {self.measurement} ' \
            f'WHERE time > {start_date} AND ' \
            f'time < {end_date} LIMIT 1 tz(\'UTC\');'

        return self.query(query)

    def interval_query(self, start_date=None, end_date=None):
        start_date = self.format_date_query(start_date)
        end_date = self.format_date_query(end_date)

        query = f'SELECT * FROM {self.measurement} ' \
            f'WHERE time > {start_date} AND time <= {end_date} tz(\'UTC\');'

        return self.query(query)

    def run(self):
        start_date = self.start_date

        end_date = start_date + self.query_interval

        while end_date <= self.now() - self.query_interval:
            for message in self.messages(start_date, end_date):
                self.tick(message['time'])

                data = json.loads(message['data'])

                if data['table'] in ['orderBookL2', 'trade']:
                    self.message(data)

            start_date += self.query_interval
            end_date += self.query_interval

    def messages(self, start, end):
        return self.interval_query(start, end).get_points(
            self.measurement)

    def save_frame(self, timestamp=None, dt=None, save_now=False):
        self.last_timestamp = timestamp

        if dt:
            date_time = dt
        else:
            date_time = self.parse_db_timestamp(self.last_timestamp)

        self._measurements += self.measurements(self.last_timestamp)

        if save_now:
            self.save_points(date_time)
        elif len(self._measurements) % self.max_cache_len == 0:
            self.save_points(date_time)

    def save_points(self, dt):
        meas = self._measurements.copy()
        self._measurements = []
        alog.info('\n' + str(self.frame_slice[:, :, :1]))
        self.write_points(meas,
                          time_precision='ms')
        alog.info(f'## meas saved for {dt}##')

    def exit(self, *args):
        self.save_frame(save_now=True)
        sys.exit(0)

    def tick(self, timestamp):
        dt = self.parse_db_timestamp(timestamp)
        if dt > self._next_tick:
            diff = self._next_tick - dt

            if diff.total_seconds() > 1:
                self._next_tick = copy(dt).replace(microsecond=0)

            self._next_tick = self._next_tick + timedelta(seconds=1)
            self.save_frame(timestamp, dt)


@click.command()
def main(**kwargs):
    orderbook = OrderBookPlayBack(depths=[21], **kwargs)
    orderbook.run()


if __name__ == '__main__':
    main()
