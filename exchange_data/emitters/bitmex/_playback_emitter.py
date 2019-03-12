import json
import sys

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
        max_cache_len: int = 60 * 60,
        **kwargs
    ):
        super().__init__(
            symbol=BitmexChannels.XBTUSD,
            subscriptions_enabled=False,
            **kwargs
        )
        self.max_cache_len = max_cache_len
        self._measurements = []
        self.measurement = 'data'
        self.start_date = start_date
        self.end_date = self.now()

    @cached_property
    def min_date(self):
        start_date = datetime.fromtimestamp(0, tz=tz.tzutc())

        result = self.oldest_frame_query(start_date, self.end_date)

        for item in result.get_points(self.measurement):
            timestamp = datetime.utcfromtimestamp(item['time'] / 1000) \
                .replace(tzinfo=tz.tzutc())
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
        start_date = start_date if start_date else self.start_date
        end_date = end_date if end_date else self.end_date

        start_date = self.format_date_query(start_date)
        end_date = self.format_date_query(end_date)

        query = f'SELECT * FROM {self.measurement} ' \
            f'WHERE time > {start_date} AND time <= {end_date} tz(\'UTC\');'

        return self.query(query)

    def run(self):
        self.start_date = self.min_date
        self.start_date.replace(microsecond=0)
        self.end_date = self.start_date + timedelta(seconds=1)

        while self.end_date <= self.now():
            for message in self.messages():
                data = json.loads(message['data'])
                if data['table'] in ['orderBookL2', 'trade']:
                    self.message(data)

            self.save_frame(self.end_date.timestamp())

            self.start_date = self.start_date + timedelta(seconds=1)
            self.end_date = self.end_date + timedelta(seconds=1)

    def messages(self):
        return self.interval_query(self.start_date, self.end_date).get_points(
            self.measurement)

    def save_frame(self, timestamp):
        self.last_timestamp = timestamp
        self._measurements += self.measurements(timestamp)

        if len(self._measurements) % self.max_cache_len == 0:
            meas = self._measurements.copy()
            self._measurements = []
            self.write_points(meas,
                              time_precision='ms')
            alog.info(f'## meas saved for {self.end_date}##')

    def exit(self, *args):
        sys.exit(0)

@click.command()
def main(**kwargs):
    orderbook = OrderBookPlayBack(depths=[21], **kwargs)
    orderbook.run()


if __name__ == '__main__':
    main()
