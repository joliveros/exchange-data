import sys
from datetime import timedelta
from multiprocessing import Process
from time import sleep

import alog
from pytimeparse.timeparse import timeparse

from exchange_data.utils import DateTimeUtils


class DateRangeSplitWorkers(object):

    def __init__(self, interval, window_size, split, max_workers, **kwargs):
        self.kwargs = kwargs
        self.max_workers = max_workers
        self._now = now = DateTimeUtils.now()
        interval_delta = timedelta(seconds=timeparse(interval))
        self.window_size = window_size
        kwargs['window_size'] = window_size
        start_date = now - interval_delta
        dates = DateTimeUtils.split_range_into_datetimes(start_date, now, split)
        self.intervals = []
        self.workers = []

        for i in range(len(dates)):
            if i < len(dates) - 1:
                self.intervals += \
                    [(dates[i + 1] - timedelta(seconds=1), dates[i])]

        # alog.info(alog.pformat([
        #     (str(interval[0]), str(interval[1])) for interval in self.intervals
        # ]))

    def run(self):
        while True:
            if len(self.workers) < self.max_workers and len(self.intervals) > 0:
                interval_dates = self.intervals.pop()
                alog.debug(f'#### ranges left {len(self.intervals)} ####')
                self.kwargs['start_date'] = interval_dates[1]
                self.kwargs['end_date'] = interval_dates[0]
                window_interval = timedelta(seconds=timeparse(self.window_size))
                interval = interval_dates[0] - interval_dates[1]

                if interval < window_interval:
                    self.kwargs['window_size'] = f'{interval.seconds}s'

                # if interval_dates[0] > self._now:
                #     raise Exception()

                alog.info(self.kwargs)

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
        alog.info(alog.pformat(args))
        alog.info(alog.pformat(kwargs))
        record = self.worker_class(*args, **kwargs)

        record.run()
