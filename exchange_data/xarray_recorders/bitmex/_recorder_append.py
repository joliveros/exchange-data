from exchange_data.cached_dataset import CachedDataset
from exchange_data.channels import BitmexChannels
from exchange_data.emitters import TimeEmitter
from pathlib import Path
from pytimeparse import parse as dateparse
from xarray import Dataset
import alog
import decimal


class RecorderAppend(CachedDataset):
    dataset: Dataset

    def __init__(
        self,
        symbol: BitmexChannels,
        save_interval='1h',
        save: bool = True,
        *args,
        **kwargs
    ):
        self.save: bool = save
        self.stopped = False
        self.save_interval = dateparse(save_interval)
        self.previous_day: decimal = TimeEmitter().previous_day

        super(RecorderAppend, self).__init__(*args, **kwargs)

        self.tick_counter = 0
        self.symbol: BitmexChannels = symbol
        self.prefix = self.symbol.value

        self.dataset = Dataset()

    @property
    def filename(self):
        return f'{self.cache_dir}/{self.prefix}_{TimeEmitter.timestamp()}.' \
            f'{self.extension}'

    @property
    def file_exists(self):
        return Path(self.filename).is_file()

    def to_netcdf(self):
        self.tick_counter += 1
        if self.tick_counter % self.save_interval == 0:
            self.write_to_file()
            self.tick_counter = 0

        elif self.stopped:
            self.write_to_file()

    def write_to_file(self):
        self.emit('save')
        if self.save:
            alog.info('### saving ###')
            with self.dataset:
                self.dataset.to_netcdf(
                    mode='w',
                    path=self.filename,
                    compute=True
                )
        self.dataset = Dataset()

    def next_day(self, timestamp):
        self.previous_day = timestamp
        self.to_netcdf()

    def dataset_frame(self, new_full_book, time):
        raise NotImplemented()

    def emit(self, event_name: str):
        raise NotImplementedError()
