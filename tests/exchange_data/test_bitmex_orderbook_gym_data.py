from exchange_data.bitmex_orderbook_gym_data import BitmexOrderBookGymData
from pytimeparse import parse as dateparse
from tests.exchange_data.fixtures import instruments, measurements
import alog
import pytest


@pytest.fixture
def book(
        instruments,
        mocker,
        tmpdir
):
    mocker.patch.object(BitmexOrderBookGymData, '_instrument_data',
                        return_value=instruments)
    total_time = '15m'
    return BitmexOrderBookGymData(
        cache_dir=tmpdir,
        overwrite=False,
        read_from_json=True,
        symbol='XBTUSD',
        total_time=total_time
    )


class TestBitmexOrderBookGymData(object):

    def test_parse_date_range_on_first_message(self, book, measurements):
        msg = book.message_strict(measurements['data'][0])

        book.read_date_range(msg)

        diff = book.date_range['end'] - book.date_range['start']

        assert diff.total_seconds() == dateparse(book.total_time)

    def test_fetch_and_save(self, book):
        book.fetch_and_save()
