from exchange_data.emitters.bitmex._playback_emitter import OrderBookPlayBack
import pytest


class TestOrderBookPlayBack(object):

    @pytest.mark.vcr()
    def test_playback(self):
        orderbook = OrderBookPlayBack()
