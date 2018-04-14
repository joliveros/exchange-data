import json

import alog
from influxdb.resultset import ResultSet

from exchange_data import Database
from exchange_data.limit_orderbook import LimitOrderBook

DATABASE_MAP = {
    'bitmex': 'orderbook'
}


class TFLimitOrderBook(LimitOrderBook, Database):
    def __init__(self, total_time: str, database: str, symbol: str, json_file=None):
        LimitOrderBook.__init__(self)
        Database.__init__(self, database_name=database)

        self.total_time = total_time
        self.symbol = symbol
        self.database = database

        if json_file is None:
            self._fetch_data()
        else:
            self._read_from_file(json_file)

    def _read_from_file(self, json_file):
        self.result_set = ResultSet(json.loads(open(json_file).read()))

    def _fetch_data(self):
        """
        fetch log data from influxdb
        :return:
        """
        query = f'SELECT * FROM orderbook WHERE time > now() - {self.total_time}'

        alog.debug(query)

        params = dict(precision='ms')

        result: ResultSet = self.query(database=self.database, query=query, epoch='ms',
                                       params=params, chunked=True)

        self.result_set = result

    def save_replay(self):
        # replay the orderbook and save at each step
        pass

    def replay(self):
        for line in self.result_set[DATABASE_MAP[self.database]]:
            self.on_message(line)

    def on_message(self, message):
        raise NotImplementedError


