from exchange_data.influxdb_orderbook import InfluxOrderBook
from pathlib import Path
from stringcase import snakecase
from tables import File

import alog
import numpy as np
import os
import tables


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

        self._file = tables.open_file(self.filename, mode='w')
        filters = tables.Filters(complevel=5, complib='blosc')
        self._storage = self.file.create_earray(
            self.file.root,
            self.storage_name,
            tables.Atom.from_dtype(
                np.dtype('Float64')),
            shape=(0, 2000, 2),
            filters=filters,
            expectedrows=1000
        )

    @property
    def storage_name(self):
        return f'{self.prefix}_{self.symbol}_{self.database}_{self.total_time}'

    @property
    def filename(self):
        return f'{self.cache_dir}/{self.storage_name}.{self.extension}'

    @property
    def file(self) -> File:
        return self._file

    @property
    def storage(self):
        return self._storage

    def file_exists(self):
        return Path(self.filename).is_file()

    def cache_directory_exists(self):
        return Path(self.cache_dir).is_dir()

    def _file_check(self):
        if self.file_check is False:
            return

        if self.overwrite:
            os.remove(self.filename)

    def fetch_and_save(self):
        raise NotImplementedError()
