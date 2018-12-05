from bitmex_websocket import Instrument
from bitmex_websocket.constants import InstrumentChannels
from datetime import datetime
from exchange_data.bitmex_orderbook_gym_data import BitmexOrderBookGymData


class LiveBitmexOrderBook(BitmexOrderBookGymData, Instrument):
    channels = [
        InstrumentChannels.trade,
        InstrumentChannels.orderBookL2
    ]

    def __init__(self, symbol: str):
        BitmexOrderBookGymData.__init__(self, symbol=symbol, overwrite=True)
        Instrument.__init__(self,
                            symbol=symbol,
                            should_auth=False,
                            channels=self.channels)

        self.on('action', self.message)

    def save_to_array(self, date: datetime):
        pass


if __name__ == '__main__':
    orderbook = LiveBitmexOrderBook('XBTUSD')
    orderbook.run_forever()
