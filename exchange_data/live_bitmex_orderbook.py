from enum import Enum
from functools import lru_cache

import requests
from bitmex_websocket import Instrument
from bitmex_websocket.constants import InstrumentChannels

from exchange_data.bitmex_orderbook import BitmexOrderBook, BitmexMessage, \
    BitmexOrder

class BitmexTickSize(Enum):
    XBTUSD = 0.01

class LiveBitmexOrderBook(BitmexOrderBook, Instrument):
    channels = [
        InstrumentChannels.trade,
        InstrumentChannels.orderBookL2
    ]

    INSTRUMENTS_URL = 'https://www.bitmex.com/api/v1/instrument?columns' \
                      '=symbol,tickSize&start=0&count=500'

    def __init__(self, symbol: str):
        BitmexOrderBook.__init__(self, symbol=symbol)
        Instrument.__init__(self,
                            symbol=symbol,
                            should_auth=False,
                            channels=self.channels)
        
        self._get_instrument_info()
        self.on('action', self.message)

    @lru_cache(maxsize=None)
    def parse_price_from_id(self, id: int):
        return ((100000000 * self.index) - id) * self.tick_size

    def _instrument_data(self):
        r = requests.get(self.INSTRUMENTS_URL)
        return r.json()

    def _get_instrument_info(self):
        all_instruments = self._instrument_data()
        instrument_data = [data for data in all_instruments
                           if data['symbol'] == self.symbol][0]
        self.index = all_instruments.index(instrument_data)

        self.tick_size = instrument_data['tickSize']

        if BitmexTickSize[self.symbol]:
            self.tick_size = BitmexTickSize[self.symbol].value

    def update_orders(self, message: BitmexMessage):
        orders = message.action.orders
        timestamp = message.timestamp

        for order in orders:
            try:
                uid = order['id']

                if 'price' not in order:
                    order['price'] = self.parse_price_from_id(uid)

                if self.order_exists(uid):
                    self.modify_order(order['id'], order['price'],
                                      quantity=order['size'],
                                      timestamp=timestamp)
                else:
                    new_order = BitmexOrder(order, message.timestamp)
                    self.process_order(new_order)

            except Exception as e:
                pass

    def symbol_index(self):

        pass


if __name__ == '__main__':
    orderbook = LiveBitmexOrderBook('XBTUSD')
    orderbook.start()
