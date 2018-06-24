from exchange_data.live_bitmex_orderbook import LiveBitmexOrderBook
from .fixtures import instruments

class TestLiveBitmexOrderBook(object):

    def test_parse_price_from_id(self, mocker, instruments):
        mocker.patch.object(LiveBitmexOrderBook, '_instrument_data',
                            lambda context: instruments)
        orderbook = LiveBitmexOrderBook(symbol='XBTUSD')
        assert orderbook.tick_size == 0.01

        uid = 8799386750

        price = orderbook.parse_price_from_id(uid)

        assert 6132.5 == price
