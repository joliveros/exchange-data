import alog
import pytest

from exchange_data.emitters.bitmex import BitmexChannels
from exchange_data.live_bitmex_orderbook import LiveBitmexOrderBook
from .fixtures import instruments, initial_orderbook_l2 # noqa; F401


class TestLiveBitmexOrderBook(object):

    @pytest.mark.vcr()
    def test_parse_initial_orderbook_state(self, initial_orderbook_l2):
        orderbook = LiveBitmexOrderBook(symbol=BitmexChannels.XBTUSD)
        message = orderbook.message(initial_orderbook_l2)

        assert message.symbol == 'XBTUSD'

        action = message.action
        assert len(action.orders) == 4
        assert action.table == 'orderBookL2'
