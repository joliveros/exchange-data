import os

import pytest

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


@pytest.fixture('module')
def measurements():
    data = [{'time': 1528681218618, 'data': '{"table": "orderBookL2", '
                                            '"action": "update", "data": [{'
                                            '"id": 8799327400, "side": "Buy", '
                                            '"size": 132658}, {"id": '
                                            '8799327650, "side": "Buy", '
                                            '"size": 19475}], "symbol": '
                                            '"XBTUSD", "timestamp": '
                                            '"2018-06-11 01:40:18.614585Z"}',
             'symbol': 'XBTUSD'},
            {'time': 1528681243802, 'data': '{"table": "orderBookL2", '
                                            '"action": '
                                 '"update", "data": [{"id": 8799298050, '
                                 '"side": "Sell", "size": 1200}, {"id": '
                                 '8799310150, "side": "Sell", "size": 3041}, '
                                 '{"id": 8799318900, "side": "Sell", "size": '
                                 '78161}, {"id": 8799324300, "side": "Sell", '
                                 '"size": 385505}, {"id": 8799333250, "side": '
                                 '"Buy", "size": 1850}, {"id": 8799334050, '
                                 '"side": "Buy", "size": 76580}, {"id": '
                                 '8799342800, "side": "Buy", "size": 16218}, '
                                 '{"id": 8799354900, "side": "Buy", "size": '
                                 '48134}, {"id": 8799370400, "side": "Buy", '
                                 '"size": 153600}], "symbol": "XBTUSD", '
                                 '"timestamp": "2018-06-11 '
                                 '01:40:43.799837Z"}', 'symbol': 'XBTUSD'}]

    return {'data': data}
