from datetime import datetime, timedelta
from dateutil import parser
from exchange_data.utils import random_date


def test_random_date():
    end = datetime.now()
    start = end - timedelta(days=365)

    rnd_datetime = random_date(end, start)

    assert type(rnd_datetime) == datetime
    assert parser.parse('2018-06-08 04:12:38.878982') != rnd_datetime
