from exchange_data.bitmex_orderbook import BitmexOrderBook
from exchange_data.emitters import Messenger, BitmexEmitterBase
from exchange_data.orderbook._ordertree import OrderTree
from multiprocessing import Lock, Process
from multiprocessing.managers import BaseManager
from time import sleep

import alog
import json


class LiveBitmexOrderBook(BitmexEmitterBase, Messenger):

    def __init__(self, symbol: str, host: str = None):
        BitmexEmitterBase.__init__(self, symbol)
        Messenger.__init__(self, host=host)
        BaseManager.register(OrderTree.__qualname__, OrderTree)
        self.manager = manager = BaseManager()

        manager.start()

        self.lock: Lock = Lock()

        self.asks = manager.OrderTree()
        self.bids = manager.OrderTree()

        self.orderbook = BitmexOrderBook(symbol=symbol)

        # self.serialize_keys: List[str] = BitmexOrderBook(symbol=symbol).__dict__.keys()

        self.on(self.channel_name, self._message)

    def _message(self, msg):
        if msg.get('type') != 'subscribe':
            data = json.loads(msg['data'])

            if data['table'] == 'orderBookL2':
                self.orderbook.emit('orderBookL2', data)

    def sub(self, lock: Lock, channel: str, asks: OrderTree, bids: OrderTree):
        self.orderbook.asks = asks
        self.orderbook.bids = bids
        super().sub(channel, lock)

    def run(self):
        sub = Process(
            target=self.sub,
            args=[self.channel_name, self.lock, self.asks, self.bids],
            daemon=True
        )
        sub.start()



if __name__ == '__main__':
    orderbook: LiveBitmexOrderBook = LiveBitmexOrderBook('XBTUSD', host='0.0.0.0')
    orderbook.run()
    sleep(10)
    alog.info(orderbook.lock)
    orderbook.lock.acquire()
    alog.info(orderbook.bids.__dict__)
    orderbook.lock.release()
