import alog

from exchange_data.utils import nice_print
from exchange_data.tf_orderbook._tf_bitmex_orderbook import TFBitmexLimitOrderBook


class TestTFBitmexLimitOrderBook(object):
    def test_instance(self):
        TFBitmexLimitOrderBook('1h', symbol='xbtusd',
                               json_file='./tests/exchange_data/data/bitmex_orderbook_log.json')

    def test_replay(self):
        orderbook = TFBitmexLimitOrderBook('1h', symbol='xbtusd',
                               json_file='./tests/exchange_data/data/bitmex_orderbook_log.json')
        # orderbook = TFBitmexLimitOrderBook('1h', symbol='xbtusd')

        orderbook.replay()

        for side in orderbook.levels(10).items():
            key, value = side
            alog.debug(key)
            levels = [{'price': level.price, 'orders': level.size} for level in value]

            alog.debug(nice_print(levels))
