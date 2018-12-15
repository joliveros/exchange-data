from datetime import datetime
from exchange_data.bitmex_orderbook_gym_data import BitmexOrderBookGymData
from exchange_data.emitters import Messenger, BitmexEmitterBase


class LiveBitmexOrderBook(BitmexEmitterBase, BitmexOrderBookGymData, Messenger):

    def __init__(self, symbol: str):
        BitmexEmitterBase.__init__(self, symbol)
        BitmexOrderBookGymData.__init__(self, symbol=symbol, overwrite=True)
        Messenger.__init__(self)

        self.on('action', self.message)

    def sub(self, **kwargs):
        super().sub(self.channel_name)

    def save_to_array(self, date: datetime):
        pass


if __name__ == '__main__':
    orderbook = LiveBitmexOrderBook('XBTUSD')
    orderbook.sub()
