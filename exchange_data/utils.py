from datetime import datetime
from enum import Enum
from random import random

from dateutil.tz import tz
from pytimeparse import parse as dateparse

import alog
import math
import tracemalloc


def roundup_to_nearest(value, interval=10.0):
    return math.ceil(value / interval) * interval


def date_plus_timestring(timestamp: int, total_time: str) -> datetime:
    total_time_ms = dateparse(total_time) * 1000
    return datetime_from_timestamp(timestamp, total_time_ms)


def datetime_from_timestamp(timestamp, total_time_ms=0):
    return datetime.fromtimestamp((total_time_ms + timestamp) / 1000)


def random_date(start, end):
    return start + random() * (end - start)


class NoValue(Enum):
    def __repr__(self):
        return '<%s.%s>' % (self.__class__.__name__, self.name)


class MemoryTracing(object):
    snapshot = None

    def __init__(self, enable_memory_tracing: bool = False, **kwargs):
        self.enable_memory_tracing = enable_memory_tracing

        if enable_memory_tracing:
            tracemalloc.start()
        else:
            self.__dict__['trace_print'] = lambda: None

    def trace_print(self):
        snapshot2 = tracemalloc.take_snapshot()
        snapshot2 = snapshot2.filter_traces((
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
            tracemalloc.Filter(False, tracemalloc.__file__)
        ))

        if self.snapshot:
            print("================================== Begin Trace:")
            top_stats = snapshot2.compare_to(self.snapshot, 'lineno', cumulative=True)
            for stat in top_stats[:10]:
                print(stat)

        self.snapshot = snapshot2


class DateTimeUtils(object):
    @staticmethod
    def now():
        return datetime.utcnow().replace(tzinfo=tz.tzutc())

    @staticmethod
    def format_date_query(value):
        return f'\'{value.astimezone(tz.tzutc()).replace(tzinfo=None)}\''

    @staticmethod
    def parse_db_timestamp(timestamp):
        return datetime.utcfromtimestamp(timestamp / 1000) \
            .replace(tzinfo=tz.tzutc())
