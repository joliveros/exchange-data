import alog
import pytest

from .fixtures import orderbook, orders
from exchange_data.orderbook import OrderBook, PriceDoesNotExistException


class TestGetVolume(object):

    def test_volume_at_price(self, orderbook: OrderBook):
        volume = orderbook.get_volume(70.0)
        assert volume == 5

    def test_raises_exception_price_does_not_exist(self, orderbook: OrderBook):
        with pytest.raises(PriceDoesNotExistException):
            orderbook.get_volume(10.0)


