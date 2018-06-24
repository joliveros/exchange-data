from exchange_data import Database
from exchange_data.orderbook import OrderBook
from influxdb.resultset import ResultSet

import alog
import exchange_data
import json
import os


class InfluxOrderBook(OrderBook, Database):
    def __init__(self, database: str, symbol: str, total_time='1d',
                 read_from_json: bool = False):
        OrderBook.__init__(self)
        Database.__init__(self, database_name=database)

        self.read_from_json = read_from_json
        self.total_time = total_time
        self.symbol = symbol
        self.database = database
        base_dir = os.path.split(os.path.dirname(exchange_data.__file__))[0]
        self.json_fixture_dir = f'{base_dir}/tests/exchange_data/data'
        self.result_set = None

    @property
    def json_fixture_filename(self):
        return f'{self.json_fixture_dir}/{self.database}_{self.total_time}.json'

    def fetch_measurements(self):
        """
        fetch log data from influxdb
        :return:
        """
        query = f'SELECT * FROM data WHERE time > now() - {self.total_time};'

        alog.debug(query)

        if self.read_from_json:
            self.result_set = self.read_from_json_fixture()
        else:
            self.result_set = self.query(database=self.database, query=query,
                                         epoch='ms', params={'precision': 'ms'},
                                         chunked=True)

    def save_json(self):
        with open(self.json_fixture_filename, 'w') as json_file:
            json.dump(self.result_set.raw, json_file)

    def read_from_json_fixture(self):
        return ResultSet(json.loads(open(self.json_fixture_filename).read()))

