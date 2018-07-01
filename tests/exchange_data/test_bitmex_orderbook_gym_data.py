import alog

from exchange_data.bitmex_orderbook_gym_data import BitmexOrderBookGymData


class TestBitmexOrderBookGymData(object):

    def test_fetch_and_save(self, mocker, tmpdir):
        book = BitmexOrderBookGymData(symbol='XBTUSD',
                                      total_time='15m', overwrite=False,
                                      cache_dir=tmpdir,
                                      read_from_json=True)

        book.fetch_and_save()
