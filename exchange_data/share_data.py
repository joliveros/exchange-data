from exchange_data.bitmex_orderbook import BitmexOrderBook
from exchange_data.emitters import Messenger, BitmexEmitterBase
from multiprocessing import Lock, Process
from multiprocessing.managers import BaseManager
from time import sleep

import alog
import json


class LiveBitmexOrderBook(BitmexEmitterBase, Messenger, BitmexOrderBook):

    def __init__(self, symbol: str, host: str = None):
        BitmexEmitterBase.__init__(self, symbol)
        Messenger.__init__(self, host=host)
        BitmexOrderBook.__init__(self, symbol=symbol)

        self.start()

        self.lock: Lock = Lock()

        self.orders = self.list()


        # self.serialize_keys: List[str] = BitmexOrderBook(symbol=symbol).__dict__.keys()

        self.on(self.channel_name, self._message)

    def _message(self, msg):
        if msg.get('type') != 'subscribe':
            data = json.loads(msg['data'])

            if data['table'] == 'orderBookL2':
                self.orderbook.emit('orderBookL2', data)

    def sub(self, **kwargs):
        sub = Process(
            target=super().sub,
            args=[self.channel_name, self.lock]
            # daemon=True
        )
        sub.start()



if __name__ == '__main__':
    orderbook: LiveBitmexOrderBook = LiveBitmexOrderBook('XBTUSD', host='0.0.0.0')
    orderbook.sub()
    sleep(10)
    orderbook.lock.acquire()
    alog.info(orderbook.orderbook.bids)
    orderbook.lock.release()
