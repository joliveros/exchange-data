from xarray import DataArray

from exchange_data.cached_dataset import CachedDataset
from exchange_data.emitters import TimeEmitter
from exchange_data.emitters.bitmex import BitmexChannels
from pytimeparse import parse as dateparse

import alog
import decimal
import xarray


class RecorderAppend(CachedDataset):
    def __init__(
        self,
        symbol: BitmexChannels,
        save_interval='1h',
        overwrite: bool = False,
        cache_dir: str = None
    ):
        self.stopped = False
        self.save_interval = dateparse(save_interval)
        self.previous_day: decimal = TimeEmitter().previous_day

        CachedDataset.__init__(self, cache_dir=cache_dir, overwrite=overwrite)

        self.tick_counter = 0
        self.symbol: BitmexChannels = symbol
        self.prefix = self.symbol.value

    @property
    def filename(self):
        return f'{self.cache_dir}/{self.prefix}_{int(self.previous_day)}.'\
            f'{self.extension}'

    def to_netcdf(self):
        self.tick_counter += 1

        if self.tick_counter % self.save_interval == 0:
            alog.info('### saving ###')
            self.append_to_file()
            self.tick_counter = 0

        elif self.stopped:
            alog.info('### saving ###')
            self.append_to_file()

    def append_to_file(self):
        if len(self.dataset.time) > 1:
            last_index: DataArray = self.dataset.time[-2:]
            dataset = self.dataset.sel(time=slice(*last_index.data))

            try:
                full_dataset = xarray.open_dataset(self.filename)
                self.resize_datasets(full_dataset, self.dataset)
                
                alog.info(full_dataset)
                alog.info(self.dataset)
                # full_dataset = xarray.concat((full_dataset, self.dataset), dim='time')
                # full_dataset.to_netcdf(self.filename, compute=True)
            except FileNotFoundError:
                self.dataset.to_netcdf(self.filename, compute=True)

    def next_day(self, timestamp):
        self.previous_day = timestamp
        self.to_netcdf()

    def resize_datasets(self, full_dataset, dataset):
        full_dataset_shape = full_dataset.orderbook.shape
        dataset_shape = dataset.orderbook.shape

        alog.info(full_dataset_shape)
        alog.info(dataset_shape)
