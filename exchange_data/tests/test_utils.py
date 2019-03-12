from datetime import datetime, timedelta
from dateutil import parser
from exchange_data.utils import random_date, DateTimeUtils
from pytimeparse.timeparse import timeparse

import alog


def test_random_date():
    end = datetime.now()
    start = end - timedelta(days=365)

    rnd_datetime = random_date(end, start)

    assert type(rnd_datetime) == datetime
    assert parser.parse('2018-06-08 04:12:38.878982') != rnd_datetime


def test_split_range_into_datetimes():
    dt1 = datetime.now()
    dt = dt1 - timedelta(seconds=timeparse('365d'))

    dts = DateTimeUtils.split_range_into_datetimes(dt, dt1, 12)

    assert dts[0] == dt
    assert dts[-1] == dt1

    for _dt in dts[1:-2]:
        assert dt < _dt < dt1
