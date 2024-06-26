import sys
from datetime import timedelta
from multiprocessing import Process
from time import sleep

import alog
from pytimeparse.timeparse import timeparse

from exchange_data import Database
from exchange_data.emitters.trading_window_emitter import TradingWindowEmitter
from exchange_data.tfrecord.dataset_query import PriceChangeRanges
from exchange_data.trading import Positions
from exchange_data.utils import DateTimeUtils


class DateRangeSplitWorkers(Database, DateTimeUtils, PriceChangeRanges):

    def __init__(
        self,
        database_name,
        volume_max,
        symbol,
        position_ratio,
        window_size,
        max_workers,
        channel_name,
        interval,
        interval_type='contiguous',
        record_window='15s',
        limit: int = 0,
        **kwargs
    ):
        DateTimeUtils.__init__(self)
        PriceChangeRanges.__init__(self)

        super().__init__(symbol=symbol, database_name=database_name, **kwargs)

        self.volume_max = volume_max
        self.database_name = database_name
        self.symbol = symbol
        self.window_size = window_size
        kwargs['window_size'] = window_size
        self.kwargs = kwargs
        self.channel_name = channel_name
        self.max_workers = max_workers
        self._now = now = DateTimeUtils.now()
        interval_delta = timedelta(seconds=timeparse(interval))
        self.start_date = now - interval_delta
        self.end_date = now
        self.intervals = []

        if interval_type == 'contiguous':
            start_date = self.start_date.replace(second=0, microsecond=0)
            intervals = int(interval_delta.total_seconds()/60)

            for i in range(intervals - 1):
                end_date = start_date + timedelta(minutes=1)
                self.intervals.append((start_date, end_date))
                start_date = start_date + timedelta(minutes=1)

        if interval_type == 'price_change':
            self.intervals = self.price_change_ranges(
                position_ratio=position_ratio,
                record_window=record_window,
                start_date=self.start_date,
                end_date=self.end_date
            )

        if interval_type == 'volatility_window':
            twindow = TradingWindowEmitter(interval=interval, group_by='2m',
                                           database_name=database_name,
                                           symbol=symbol)
            twindow.next_intervals()
            alog.info(twindow.intervals)
            self.intervals = twindow.intervals

        if limit > 0:
            self.intervals = self.intervals[(limit * -1):]

        self.workers = []

    def run(self):
        while True:
            if len(self.workers) < self.max_workers and len(self.intervals) > 0:
                interval_dates = self.intervals.pop()
                # alog.info(f'#### ranges left {len(self.intervals)} ####')
                self.kwargs['start_date'] = interval_dates[0]
                self.kwargs['end_date'] = interval_dates[1]
                self.kwargs['directory_name'] = self.directory_name
                self.kwargs['database_name'] = self.database_name
                self.kwargs['symbol'] = self.symbol
                self.kwargs['volume_max'] = self.volume_max

                window_interval = \
                    timedelta(seconds=timeparse(self.window_size))

                interval = interval_dates[1] - interval_dates[0]

                if interval < window_interval:
                    self.kwargs['window_size'] = f'{interval.seconds}s'

                worker = Process(
                    target=self.worker,
                    args=(),
                    kwargs=self.kwargs
                )

                worker.start()
                alog.debug(worker)

                self.workers.append(worker)

            if len(self.intervals) == 0 and len(self.workers) == 0:
                # sys.exit(0)
                return

            self.workers = [w for w in self.workers if w.is_alive()]

            sleep(1)

    def worker(self, *args, **kwargs):
        record = self.worker_class(*args, **kwargs)

        record.run()
