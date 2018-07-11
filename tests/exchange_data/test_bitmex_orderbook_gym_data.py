import alog

from exchange_data.bitmex_orderbook_gym_data import BitmexOrderBookGymData
from mock import MagicMock
from pytimeparse import parse as dateparse
from tests.exchange_data.fixtures import instruments


class TestBitmexOrderBookGymData(object):

    def test_fetch_and_save(self, mocker, tmpdir, instruments):
        mocker.patch.object(BitmexOrderBookGymData, '_instrument_data',
                            lambda context: instruments)

        book = BitmexOrderBookGymData(symbol='XBTUSD',
                                      total_time='1h', overwrite=False,
                                      cache_dir=tmpdir,
                                      read_from_json=True)

        book.fetch_and_save()
