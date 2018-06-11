from pathlib import Path
from stringcase import snakecase

import alog
import h5py
import os

from exchange_data.influxdb_orderbook import InfluxOrderBook


class Hdf5OrderBook(InfluxOrderBook):
    def __init__(self,
                 database: str,
                 symbol: str,
                 cache_dir: str = None,
                 file_check=True,
                 overwrite: bool = False,
                 read_from_json: bool = False,
                 total_time='1d'
                 ):
        super().__init__(database=database, symbol=symbol,
                         total_time=total_time, read_from_json=read_from_json)

        self.file_check = file_check
        self.overwrite = overwrite

        if cache_dir is None:
            cache_dir = f'{Path.home()}/.exchange-data'

        self.cache_dir = cache_dir
        self.prefix = snakecase(self.__class__.__name__)
        self.extension = 'hdf5'

        if not self.cache_directory_exists():
            os.makedirs(self.cache_dir, exist_ok=True)

        self._file_check()

    @property
    def filename(self):
        return f'{self.cache_dir}/{self.prefix}_{self.symbol}_' \
               f'{self.database}_{self.total_time}.{self.extension}'

    def file_exists(self):
        return Path(self.filename).is_file()

    def cache_directory_exists(self):
        return Path(self.cache_dir).is_dir()

    def file(self):
        return h5py.File(self.filename)

    def _file_check(self):
        if self.file_check is False:
            return

        if not self.file_exists():
            self.fetch_and_save()

    def fetch_and_save(self):
        raise NotImplementedError()
