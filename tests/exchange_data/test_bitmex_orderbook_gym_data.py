from exchange_data.bitmex_orderbook_gym_data import BitmexOrderBookGymData
from pytimeparse import parse as dateparse
from tests.exchange_data.fixtures import instruments, measurements

import alog
import pytest


@pytest.fixture
def book(tmpdir):
    total_time = '5m'

    return BitmexOrderBookGymData(
        cache_dir=tmpdir,
        overwrite=False,
        read_from_json=False,
        symbol='XBTUSD',
        total_time=total_time
    )


class TestBitmexOrderBookGymData(object):

    @pytest.mark.vcr()
    def test_parse_date_range_on_first_message(
            self,
            book: BitmexOrderBookGymData,
            measurements
    ):
        msg = book.message(measurements['data'][0])

        book.read_date_range(msg)

        diff = book.date_range['end'] - book.date_range['start']

        assert diff.total_seconds() == dateparse(book.total_time)

    @pytest.mark.vcr()
    def test_fetch_and_save(self, book: BitmexOrderBookGymData):
        book.fetch_and_save()
