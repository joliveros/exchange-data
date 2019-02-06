import math
import tracemalloc
from datetime import datetime
from enum import Enum

from pytimeparse import parse as dateparse


def roundup_to_nearest(value, interval=10.0):
    return math.ceil(value / interval) * interval


def date_plus_timestring(timestamp: int, total_time: str) -> datetime:
    total_time_ms = dateparse(total_time) * 1000
    return datetime_from_timestamp(timestamp, total_time_ms)


def datetime_from_timestamp(timestamp, total_time_ms=0):
    return datetime.fromtimestamp((total_time_ms + timestamp) / 1000)


snapshot = None


def trace_print():
    global snapshot
    snapshot2 = tracemalloc.take_snapshot()
    snapshot2 = snapshot2.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
        tracemalloc.Filter(False, tracemalloc.__file__)
    ))

    if snapshot is not None:
        print("================================== Begin Trace:")
        top_stats = snapshot2.compare_to(snapshot, 'lineno', cumulative=True)
        for stat in top_stats[:10]:
            print(stat)
    snapshot = snapshot2

class NoValue(Enum):
    def __repr__(self):
        return '<%s.%s>' % (self.__class__.__name__, self.name)