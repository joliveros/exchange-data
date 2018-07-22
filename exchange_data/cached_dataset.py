from pathlib import Path
from stringcase import snakecase
from xarray import Dataset, open_dataset
import alog
import os


class CachedDataset(object):
    dataset: Dataset

    def __init__(self,
                 cache_dir: str = None,
                 overwrite: bool = False
                 ):
        self.dataset = None
        self.overwrite = overwrite

        if cache_dir is None:
            cache_dir = f'{Path.home()}/.exchange-data'

        self.cache_dir = cache_dir
        self.prefix = snakecase(self.__class__.__name__)
        self.extension = 'nc'

        if not self.cache_directory_exists():
            os.makedirs(self.cache_dir, exist_ok=True)

        self._file_check()

        if self.file_exists():
            self.dataset = open_dataset(self.filename)
        else:
            self.dataset = Dataset()

    @property
    def storage_name(self):
        return f'{self.prefix}'

    @property
    def filename(self):
        return f'{self.cache_dir}/{self.storage_name}.{self.extension}'

    @property
    def storage(self):
        return self._storage

    def file_exists(self):
        return Path(self.filename).is_file()

    def cache_directory_exists(self):
        return Path(self.cache_dir).is_dir()

    def _file_check(self):
        if self.overwrite:
            os.remove(self.filename)

    def save(self):
        self.dataset.to_netcdf(self.filename)

    def fetch_and_save(self):
        raise NotImplementedError()
