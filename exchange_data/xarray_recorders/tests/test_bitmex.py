from exchange_data.emitters import BitmexChannels
from exchange_data.xarray_recorders import BitmexXArrayRecorder

import alog
import pytest


class TestBitmexXArrayRecorder(object):

    @pytest.mark.vcr()
    def test_init_(self, mocker, tmpdir):
        mocker.patch('exchange_data.emitters.messenger.Redis')
        mocker.patch('exchange_data.cached_dataset.CachedDataset.file_exists',
                     return_value=False)

        recorder = BitmexXArrayRecorder(symbol=BitmexChannels.XBTUSD,
                                        cache_dir=tmpdir)

    @pytest.mark.vcr()
    def test_file_name_override(self, mocker):
        mocker.patch('exchange_data.emitters.messenger.Redis')
        mocker.patch('exchange_data.cached_dataset.CachedDataset.file_exists',
                     return_value=False)

        recorder = BitmexXArrayRecorder(symbol=BitmexChannels.XBTUSD)

        assert recorder.filename == f'{recorder.cache_dir}/{recorder.prefix}_{int(recorder.previous_day)}.{recorder.extension}'
