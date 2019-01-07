from exchange_data.cached_dataset import CachedDataset


class TestCachedDataset(object):
    def test_create_new_and_reopen(self, mocker, tmpdir):
        cd = CachedDataset(overwrite=True, cache_dir=tmpdir)
        cd.to_netcdf()

