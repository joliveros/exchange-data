import os

from exchange_data.utils import roundup_to_nearest


class TestUtils(object):
    def test_roundup_to_nearest_ten(self):
        result = roundup_to_nearest(1.0, 10.0)

        assert result == 10.0

    def test_roundup_to_nearest_twenty(self):
        result = roundup_to_nearest(1.0, 20.0)

        assert result == 20.0


def datafile_name(name):
    path = f'./data/{name}.json'
    dir = os.path.join(os.path.dirname(__file__))
    path = os.path.join(dir, path)
    return path
