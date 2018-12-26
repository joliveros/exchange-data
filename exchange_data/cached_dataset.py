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
        self.should_overwrite = overwrite

        if cache_dir is None:
            cache_dir = f'{Path.home()}/.exchange-data'

        self.cache_dir = cache_dir
        self.prefix = snakecase(self.__class__.__name__)
        self.extension = 'nc'

        if not self.cache_directory_exists():
            os.makedirs(self.cache_dir, exist_ok=True)

        self.overwrite()

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

    def overwrite(self):
        if self.should_overwrite and self.file_exists():
            os.remove(self.filename)

    def to_netcdf(self):
        self.dataset.to_netcdf(self.filename, compute=True)
