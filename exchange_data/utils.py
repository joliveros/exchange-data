from abc import ABC
from datetime import datetime, timedelta
from dateutil import parser
from dateutil.tz import tz
from enum import Enum
from pyee import EventEmitter
from pytimeparse import parse as dateparse
from random import random

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


class DateTimeUtils(ABC):
    def __init__(self, **kwargs):

        pass


    @staticmethod
    def remove_seconds(value):
        if value:
            return value.replace(second=0, microsecond=0)
        return value

    @staticmethod
    def now():
        return datetime.utcnow().replace(tzinfo=tz.tzutc())

    @staticmethod
    def format_date_query(value):
        return f'\'{value.astimezone(tz.tzutc()).replace(tzinfo=None)}\''

    @staticmethod
    def parse_timestamp(timestamp, tz=tz.tzlocal()):
        return datetime.utcfromtimestamp(timestamp)\
            .replace(tzinfo=tz)

    @staticmethod
    def parse_db_timestamp(timestamp):
        return datetime.utcfromtimestamp(timestamp / 1000) \
            .replace(tzinfo=tz.tzutc())

    @staticmethod
    def parse_datetime_str(value):
        return parser.parse(value).replace(tzinfo=tz.tzutc())

    @staticmethod
    def split_range_into_datetimes(
        dt: datetime,
        dt1: datetime,
        num_intervals: int
    ):
        delta: timedelta = dt1 - dt
        interval = round(delta.total_seconds() / num_intervals)
        interval_delta = timedelta(seconds=interval)
        last_dt = None

        results = []
        results.append(dt)

        for i in range(num_intervals):
            if last_dt is None:
                last_dt = dt
            last_dt = last_dt + interval_delta
            results.append(last_dt)

        return results


class Base(object):
     def __init__(self, **kwargs):
        pass


class EventEmitterBase(ABC):
    event_emitter_class = EventEmitter

    def __init__(self, **kwargs):
        if 'event_emitter' not in self.__dict__:
            self.event_emitter = self.event_emitter_class()

        self.on('error', self.raise_error)

        try:
            super().__init__(**kwargs)
        except:
            pass

    def raise_error(self, error):
        alog.info(error)
        raise Exception(error)

    def on(self, event, f=None):
        return self.event_emitter.on(event, f)

    def emit(self, *args, **kwargs):
        return self.event_emitter.emit(*args, **kwargs)
