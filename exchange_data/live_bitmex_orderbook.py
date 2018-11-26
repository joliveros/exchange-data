from bitmex_websocket import Instrument
from bitmex_websocket.constants import InstrumentChannels
from exchange_data.bitmex_orderbook import BitmexOrderBook


class LiveBitmexOrderBook(BitmexOrderBook, Instrument):
    channels = [
        InstrumentChannels.trade,
        InstrumentChannels.orderBookL2
    ]

    def __init__(self, symbol: str):
        BitmexOrderBook.__init__(self, symbol=symbol)
        Instrument.__init__(self,
                            symbol=symbol,
                            should_auth=False,
                            channels=self.channels)

        self.on('action', self.message)


if __name__ == '__main__':
    orderbook = LiveBitmexOrderBook('XBTUSD')
    orderbook.start()
