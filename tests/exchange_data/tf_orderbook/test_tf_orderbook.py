from exchange_data.tf_orderbook._tf_orderbook import TFLimitOrderBook


class TestTFLimitOrderBook(object):
    def test_instance(self):
        TFLimitOrderBook('1h', 'bitmex', 'xbtusd',
                         json_file='./tests/exchange_data/data/bitmex_orderbook_log.json')


