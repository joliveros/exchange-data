import json

import alog
from influxdb.resultset import ResultSet

from exchange_data import Database
from exchange_data.limit_orderbook import LimitOrderBook


class TFLimitOrderBook(LimitOrderBook, Database):
    def __init__(self, database: str, symbol: str, total_time='1d',
                 save_json=True, json_file=None):
        LimitOrderBook.__init__(self)
        Database.__init__(self, database_name=database)

        self.save_json = save_json
        self.total_time = total_time
        self.symbol = symbol
        self.database = database

        if json_file is None:
            self._fetch_measurements()
        else:
            self._read_from_file(json_file)

        if self.save_json:
            self.save_as_json()

    def save_as_json(self):
        with open(f'./tests/exchange_data/data/{self.database}.json', 'w') as json_file:
            json.dump(self.result_set.raw, json_file)

    def _read_from_file(self, json_file):
        self.result_set = ResultSet(json.loads(open(json_file).read()))

    def _fetch_measurements(self):
        """
        fetch log data from influxdb
        :return:
        """
        # query = f'SELECT * FROM /.*/ WHERE time > now() - ' \
        #         f'{self.total_time};'

        query = f'SELECT merge(*) FROM /.*/ WHERE time > now() - ' \
                f'{self.total_time};'
        alog.debug(query)

        params = dict(precision='ms')

        result: ResultSet = self.query(database=self.database, query=query, epoch='ms',
                                       params=params, chunked=True)
        alog.debug(result)
        self.result_set = result

    def save_replay(self):
        # replay the orderbook and save at each step
        pass

    def replay(self):
        for key, _ in self.result_set.keys():
            alog.debug(key)



    def on_message(self, message):
        raise NotImplementedError


