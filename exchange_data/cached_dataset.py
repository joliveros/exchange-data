from pathlib import Path
from stringcase import snakecase
from xarray import Dataset, open_dataset

import alog
import os


class CachedDataset(object):
    def __init__(self,
                 cache_dir: str = None,
                 overwrite: bool = False,
                 *args,
                 **kwargs
                 ):

        alog.info(cache_dir)

        if cache_dir is None:
            cache_dir = f'{Path.home()}/.exchange-data'

        self.cache_dir = cache_dir
        self.prefix = snakecase(self.__class__.__name__)
        self.extension = 'nc'

        self.dataset = Dataset()

        if not self.cache_directory_exists():
            os.makedirs(self.cache_dir, exist_ok=True)

    @property
    def filename(self):
        return f'{self.cache_dir}/{self.storage_name}.{self.extension}'

    def cache_directory_exists(self):
        return Path(self.cache_dir).is_dir()

    def to_netcdf(self):
        raise NotImplemented()
