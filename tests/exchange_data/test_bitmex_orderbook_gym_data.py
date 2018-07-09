import alog

from exchange_data.bitmex_orderbook_gym_data import BitmexOrderBookGymData
from tests.exchange_data.fixtures import instruments


class TestBitmexOrderBookGymData(object):

    def test_fetch_and_save(self, mocker, tmpdir, instruments):
        mocker.patch.object(BitmexOrderBookGymData, '_instrument_data',
                            lambda context: instruments)

        book = BitmexOrderBookGymData(symbol='XBTUSD',
                                      total_time='15m', overwrite=False,
                                      cache_dir=tmpdir,
                                      read_from_json=True)

        book.fetch_and_save()

    def test_fetch_and_save_30_second_interval(self, mocker, tmpdir,
                                               instruments):
        mocker.patch.object(BitmexOrderBookGymData, '_instrument_data',
                            lambda context: instruments)
        book = BitmexOrderBookGymData(symbol='XBTUSD',
                                      total_time='15m', overwrite=False,
                                      cache_dir=tmpdir,
                                      read_from_json=True,
                                      interval='15s')

        book.fetch_and_save()
