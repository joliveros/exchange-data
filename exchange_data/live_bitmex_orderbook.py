from multiprocessing import Lock
from typing import List

import alog

from exchange_data.bitmex_orderbook import BitmexOrderBook
from exchange_data.emitters import Messenger, BitmexEmitterBase

import json


class LiveBitmexOrderBook(BitmexEmitterBase, BitmexOrderBook, Messenger):

    def __init__(self, symbol: str, host: str = None):
        BitmexEmitterBase.__init__(self, symbol)
        BitmexOrderBook.__init__(self, symbol=symbol)
        Messenger.__init__(self, host=host)
        self.serialize_keys: List[str] = BitmexOrderBook(symbol=symbol).__dict__.keys()

        self.on(self.channel_name, self._message)
        self.on('orderBookL2', self.message)

    def _message(self, msg):
        if msg.get('type') != 'subscribe':
            data = json.loads(msg['data'])

            if data['table'] == 'orderBookL2':
                self.emit('orderBookL2', data)

    def sub(self, lock: Lock, channel: str = None):
        if channel is None:
            channel = self.channel_name

        super().sub(channel=channel, lock=lock)


if __name__ == '__main__':
    orderbook = LiveBitmexOrderBook('XBTUSD', host='0.0.0.0')
    orderbook.sub()
