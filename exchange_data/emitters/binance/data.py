#!/usr/bin/env python
import json

from binance.client import Client
from binance.websockets import BinanceSocketManager
from exchange_data.emitters import Messenger

import alog
import click
import signal


class DataEmitter(Messenger):
    def __init__(self, symbol, **kwargs):
        super().__init__(**kwargs)
        self.symbol = symbol
        symbol_low = symbol.lower()

        self.socketManager = BinanceSocketManager(Client(), user_timeout=60)
        depth_channel = f'{symbol_low}@depth@100ms'
        self.socketManager.start_multiplex_socket([depth_channel], self.message)

    def message(self, data): # alog.info(alog.pformat(data['data']))
        self.publish(self.symbol, json.dumps(data['data']))

    def start(self):
        self.socketManager.start()



@click.command()
@click.argument('symbol', type=str)
def main(**kwargs):
    emitter = DataEmitter(**kwargs)
    emitter.start()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda: exit(0))
    signal.signal(signal.SIGTERM, lambda: exit(0))
    main()
