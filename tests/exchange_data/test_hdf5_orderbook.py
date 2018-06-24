from exchange_data.hdf5_orderbook import Hdf5OrderBook
from tests.exchange_data.fixtures import measurements


class TestHdf5OrderBook(object):

    def test_fetches_and_writes_data_to_file(self, mocker, measurements,
                                             tmpdir):
        mocker.patch('exchange_data.hdf5_orderbook.InfluxOrderBook'
                     '.fetch_measurements',
                     return_value=measurements)

        book = Hdf5OrderBook(database='bitmex', symbol='XBTUSD',
                             total_time='15m', overwrite=True,
                             cache_dir=tmpdir, file_check=False)

