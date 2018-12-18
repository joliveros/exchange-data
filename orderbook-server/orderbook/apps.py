import multiprocessing

import alog as alog
from django.apps import AppConfig

from exchange_data.live_bitmex_orderbook import LiveBitmexOrderBook


class OrderbookConfig(AppConfig):
    name = 'orderbook'
    orderbook: LiveBitmexOrderBook = None
    orderbook_lock = None

    def ready(self):
        self.orderbook = LiveBitmexOrderBook('XBTUSD', host='0.0.0.0')
        self.orderbook_lock = multiprocessing.Lock()

        background_process = multiprocessing.Process(
            name='orderbook',
            target=self.orderbook.sub,
            args=(self.orderbook_lock,),
            daemon=True
        )

        background_process.start()

