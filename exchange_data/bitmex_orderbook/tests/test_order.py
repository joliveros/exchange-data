import alog

from exchange_data.bitmex_orderbook import BitmexOrder


class TestBitmexOrderObject(object):
    def test_bitmex_order(self):
        order = BitmexOrder(
            instrument_index=88,
            order_data={
                'id': 8799596350,
                'side': 'Buy',
                'size': 300,
                'symbol': 'XBTUSD'
            },
            tick_size=0.5,
            timestamp=1546747235.2815228
        )

        assert order.price == 4036.5
