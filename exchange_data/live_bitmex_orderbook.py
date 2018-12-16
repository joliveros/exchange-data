from datetime import datetime
from exchange_data.bitmex_orderbook_gym_data import BitmexOrderBookGymData
from exchange_data.emitters import Messenger, BitmexEmitterBase

import json


class LiveBitmexOrderBook(BitmexEmitterBase, BitmexOrderBookGymData, Messenger):

    def __init__(self, symbol: str, host: str = None):
        BitmexEmitterBase.__init__(self, symbol)
        BitmexOrderBookGymData.__init__(self, symbol=symbol, overwrite=True)
        Messenger.__init__(self, host=host)

        self.on(self.channel_name, self._message)
        self.on('action', self.message)

    def _message(self, msg):
        if msg.get('type') != 'subscribe':
            data = json.loads(msg['data'])
            self.emit('action', data)

    def sub(self, **kwargs):
        super().sub(self.channel_name)

    def save_to_array(self, date: datetime):
        pass


if __name__ == '__main__':
    orderbook = LiveBitmexOrderBook('XBTUSD', host='0.0.0.0')
    orderbook.sub()
