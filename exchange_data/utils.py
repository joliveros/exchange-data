import math
from datetime import datetime
from pytimeparse import parse as dateparse


def roundup_to_nearest(value, interval=10.0):
    return math.ceil(value / interval) * interval


def date_plus_timestring(timestamp: int, total_time: str) -> datetime:
    total_time_ms = dateparse(total_time) * 1000
    return datetime.fromtimestamp((total_time_ms + timestamp) / 1000)