import alog

from exchange_data.utils import nice_print
from exchange_data.tf_orderbook._tf_bitmex_orderbook import TFBitmexLimitOrderBook


class TestTFBitmexLimitOrderBook(object):
    def test_instance(self):
        TFBitmexLimitOrderBook('1h', symbol='xbtusd',
                               json_file='./tests/exchange_data/data/bitmex.json')

    def test_replay(self):
        # orderbook = TFBitmexLimitOrderBook(symbol='xbtusd',
        #                                    json_file='./tests/exchange_data/data/bitmex.json')

        orderbook = TFBitmexLimitOrderBook(total_time='12h', symbol='xbtusd', save_json=True)

        orderbook.replay()

        levels = orderbook.levels_by_price(100)
        alog.debug(nice_print({
            'ask': orderbook.best_ask.price,
            'bid': orderbook.best_bid.price
        }))
        alog.debug(nice_print(levels))
