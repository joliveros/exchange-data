from exchange_data.cached_dataset import CachedDataset
from exchange_data.emitters import TimeEmitter
from exchange_data.emitters.bitmex import BitmexChannels
from pytimeparse import parse as dateparse
from xarray import Dataset

import alog
import decimal


class BitmexXArrayRecorder(CachedDataset):
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
            alog.debug('### saving ###')
            super().to_netcdf()
            self.tick_counter = 0
        elif self.stopped:
            alog.info('### saving ###')
            super().to_netcdf()

    def next_day(self, timestamp):
        self.previous_day = timestamp
        self.to_netcdf()
        self.dataset = Dataset()
