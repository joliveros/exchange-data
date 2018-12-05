import alog
import pytest

from exchange_data.live_bitmex_orderbook import LiveBitmexOrderBook
from .fixtures import instruments, initial_orderbook_l2 # noqa; F401


class TestLiveBitmexOrderBook(object):

    @pytest.mark.vcr()
    def test_parse_price_from_id(self):
        orderbook = LiveBitmexOrderBook(symbol='XBTUSD')
        assert orderbook.tick_size == 0.5

        uid = 8799386750

        price = orderbook.parse_price_from_id(uid)

        assert 306625.0 == price

    @pytest.mark.vcr()
    def test_parse_initial_orderbook_state(self, initial_orderbook_l2):
        orderbook = LiveBitmexOrderBook(symbol='XBTUSD')
        message = orderbook.message(initial_orderbook_l2)

        assert message.symbol == 'XBTUSD'

        action = message.action
        assert len(action.orders) == 4
        assert action.table == 'orderBookL2'
