import sys
from datetime import timedelta
from multiprocessing import Process
from time import sleep

import alog
from pytimeparse.timeparse import timeparse

from exchange_data import Database
from exchange_data.tfrecord.dataset_query import PriceChangeRanges
from exchange_data.trading import Positions
from exchange_data.utils import DateTimeUtils


class DateRangeSplitWorkers(Database, DateTimeUtils, PriceChangeRanges):

    def __init__(
        self,
        window_size,
        max_workers,
        channel_name,
        interval,
        side,
        record_window='15s',
        limit: int = 0,
        **kwargs
    ):
        DateTimeUtils.__init__(self)
        PriceChangeRanges.__init__(self)

        super().__init__(**kwargs)

        self.window_size = window_size
        kwargs['window_size'] = window_size
        kwargs['side'] = side
        self.kwargs = kwargs
        self.channel_name = channel_name
        self.max_workers = max_workers
        self._now = now = DateTimeUtils.now()
        interval_delta = timedelta(seconds=timeparse(interval))
        self.start_date = now - interval_delta
        self.end_date = now

        self.intervals = self.price_change_ranges(
            side=side,
            record_window=record_window,
            start_date=self.start_date,
            end_date=self.end_date
        )

        if limit > 0:
            self.intervals = self.intervals[(limit * -1):]

        self.workers = []

    def run(self):
        while True:
            if len(self.workers) < self.max_workers and len(self.intervals) > 0:
                interval_dates = self.intervals.pop()
                alog.debug(f'#### ranges left {len(self.intervals)} ####')
                self.kwargs['start_date'] = interval_dates[0]
                self.kwargs['end_date'] = interval_dates[1]
                self.kwargs['directory_name'] = self.directory_name

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
