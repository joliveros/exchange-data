import alog
from bitmex_websocket import Instrument
from bitmex_websocket.constants import InstrumentChannels

from exchange_data.bitmex_orderbook import BitmexOrderBook
from exchange_data.orderbook import OrderBook


class LiveBitmexOrderBook(BitmexOrderBook, OrderBook, Instrument):
    channels = [
        # InstrumentChannels.quote,
        # InstrumentChannels.trade,
        InstrumentChannels.orderBookL2
    ]

    def __init__(self, symbol: str, total_time='1d'):
        BitmexOrderBook.__init__(self, symbol=symbol, total_time=total_time)
        OrderBook.__init__(self)
        Instrument.__init__(self,
                            symbol=symbol,
                            should_auth=False,
                            channels=self.channels)

        self.on('action', self.message)


if __name__ == '__main__':
    orderbook = LiveBitmexOrderBook('XBTUSD')
    orderbook.start()
