from datetime import timedelta
from exchange_data.emitters.bitmex._playback_emitter import OrderBookPlayBack

import alog
import pytest


class TestOrderBookPlayBack(object):

    # @pytest.mark.vcr(record_mode='all')
    def test_playback(self):
        orderbook = OrderBookPlayBack(query_interval=2, depths=[21])

        orderbook.run()

    def test_get_empty_ranges(self):
        orderbook = OrderBookPlayBack(query_interval=2, depths=[21])

        orderbook.get_empty_ranges()
