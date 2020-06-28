from cached_property import cached_property
from copy import copy
from datetime import datetime, timedelta
from dateutil.tz import tz
from exchange_data.channels import BitmexChannels
from exchange_data.emitters.bitmex import BinanceOrderBookEmitter
from exchange_data.orderbook.exceptions import PriceDoesNotExistException
from exchange_data.streamers._bitmex import BitmexOrderBookChannels
from exchange_data.utils import DateTimeUtils
from multiprocessing import Process, Pool
from pytimeparse.timeparse import timeparse
from time import sleep

import alog
import click
import json
import sys


class OrderBookPlayBack(BinanceOrderBookEmitter, DateTimeUtils):
    def __init__(
        self,
        start_date=None,
        end_date=None,
        min_date=None,
        max_cache_len: int = 60 * 45,
        query_interval: int = 60 * 24,
        save_delay='15m',
        **kwargs
    ):
        super().__init__(
            symbol=BitmexChannels.XBTUSD,
            subscriptions_enabled=False,
            **kwargs
        )

        self._min_date = min_date

        self.save_delay = timeparse(save_delay)
        self.query_interval = timedelta(minutes=query_interval)
        self.query_interval_s = self.query_interval.total_seconds()
        self.max_cache_len = max_cache_len
        self._measurements = []
        self.measurement = 'data'
        self.tick_delta = timedelta(seconds=1)
        self.tick_delta_s = self.tick_delta.total_seconds()

        if start_date is None:
            self.start_date = self.min_date
        elif start_date < self.min_date:
            self.start_date = self.min_date
        else:
            self.start_date = start_date

        self.start_date = self.start_date.replace(microsecond=0)
        self.start_save = copy(self.start_date)
        self.start_date = self.start_date - \
                          timedelta(seconds=self.save_delay)
        self._next_tick = self.start_date

        if end_date:
            self.end_date = end_date
        else:
            self.end_date = self.now()

    @cached_property
    def min_date(self):
        if self._min_date:
            return self._min_date

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
        end_date_max = self.end_date + timedelta(seconds=15)

        while True:
            for message in self.messages(start_date, end_date):
                data = json.loads(message['data'])
                self.last_timestamp = self.parse_db_timestamp(message['time'])

                self.tick()

                if data['table'] in ['orderBookL2', 'trade']:
                    try:
                        self.message(data)
                    except PriceDoesNotExistException:
                        pass

            start_date += self.query_interval
            end_delta = self.end_date - end_date
            diff = end_delta.total_seconds()

            if self.query_interval_s > diff:
                end_date += timedelta(seconds=diff + 15)
            else:
                end_date += self.query_interval

            if end_date > end_date_max or start_date > end_date_max:
                self.save_points()
                break

    def messages(self, start, end):
        return self.interval_query(start, end).get_points(
            self.measurement)

    @property
    def orderbook_size(self):
        return len(self.asks) + len(self.bids)

    def save_frame(self, **kwargs):
        if self._next_tick >= self.start_save and self.orderbook_size > 0:
            self._measurements += self.measurements(self._next_tick)

            if len(self._measurements) > self.max_cache_len:
                self.save_points()
        else:
            self._measurements = []

    def save_points(self):
        if self.frame_slice is not None:
            alog.debug('\n' + str(self.frame_slice[:, :, :1]))

        self.write_points(self._measurements)

        self._measurements = []

    def exit(self, *args):
        sys.exit(0)

    def tick(self):
        while self.last_timestamp > self._next_tick:
            diff = (self._next_tick - self.last_timestamp).total_seconds()

            if diff < 0:
                self.save_frame()
                self._next_tick = self._next_tick + self.tick_delta

    def get_empty_ranges(
        self,
        min_count,
        interval,
        **kwargs
    ):
        range_lists = [self._get_empty_ranges(depth, min_count, interval, **kwargs) for depth in self.depths]
        return [range for range_list in range_lists for range in range_list]

    def _get_empty_ranges(
        self,
        depth,
        min_count=None,
        interval=None,
        start_date=None,
        end_date=None
    ):
        if start_date is None:
            start_date = self.format_date_query(self.min_date)
        else:
            start_date = self.format_date_query(start_date)

        if end_date is None:
            end_date = self.format_date_query(self.now())
        else:
            end_date = self.format_date_query(end_date)

        if interval is None:
            interval = '10d'

        if min_count is None:
            min_count = timeparse('10d')

        meas_name = self.channel_for_depth(depth)
        query = f'SELECT COUNT(*) FROM {meas_name} ' \
            f'WHERE time > {start_date} AND time <= {end_date} ' \
            f'GROUP BY time({interval}) tz(\'UTC\');'

        incomplete_ranges = [(depth, item)
                             for item in self.query(query).get_points(meas_name)
                             if item['count_data'] < min_count]

        dts = [(range[0], self.parse_db_timestamp(range[1]['time']))
               for range in incomplete_ranges]

        dt_ranges = [
            (dt[0], dt[1], dt[1] + timedelta(seconds=timeparse(interval) - 1))
            for dt in dts
        ]
        return dt_ranges


@click.command()
@click.option('--max-workers', '-w', type=int, default=12)
@click.option('--max-count', '-c', type=int, default=10000)
@click.option('--group-interval', '-g', type=str, default='5d')
@click.option('--query-interval', '-q', type=int, default=60*24)
@click.option('--max-cache-len', type=int, default=60*60*3)
def main(max_workers, max_count, group_interval, **kwargs):
    ranges = OrderBookPlayBack(depths=[21], **kwargs)\
        .get_empty_ranges(max_count, group_interval)

    ranges.reverse()

    def replay(depth, start_date, end_date, **kwargs):
        orderbook = OrderBookPlayBack(
            depths=[depth],
            start_date=start_date,
            end_date=end_date,
            **kwargs
        )
        return orderbook.run()

    workers = []

    while True:
        if len(workers) < max_workers and len(ranges) > 0:
            args = ranges.pop()
            alog.debug(f'#### ranges left {len(ranges)} ####')
            worker = Process(target=replay, args=args, kwargs=kwargs)
            worker.start()
            alog.debug(worker)
            workers.append(worker)
        if len(ranges) == 0 and len(workers) == 0:
            sys.exit(0)

        workers = [w for w in workers if w.is_alive()]

        sleep(1)


if __name__ == '__main__':
    main()
