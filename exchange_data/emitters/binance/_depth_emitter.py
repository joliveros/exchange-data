#!/usr/bin/env python
import alog
from binance.client import Client
from binance.depthcache import DepthCacheManager
from datetime import timedelta, datetime
from exchange_data.emitters import Messenger
from pytimeparse.timeparse import timeparse

import click
import json
import numpy as np
import os
import signal
import time


class DepthEmitter(Messenger):
    def __init__(self, symbol, **kwargs):
        super().__init__(**kwargs)
        self.symbol = symbol
        self.timeout = None
        self.clear_timeout()
        self.on('clear_timeout', self.clear_timeout)

        self.depthCache = DepthCacheManager(Client(), symbol=symbol,
                                            callback=self.message, limit=5000,
                                            refresh_interval=timeparse('10m'))

        while self.timeout > datetime.now():
            time.sleep(0.1)

        self.exit()

    def exit(self):
        os.kill(os.getpid(), signal.SIGKILL)

    def message(self, depthCache):
        if depthCache is None:
            self.exit()

        asks = np.expand_dims(np.asarray(depthCache.get_asks()), axis=0)
        bids = np.expand_dims(np.asarray(depthCache.get_bids()), axis=0)
        ask_levels = asks.shape[1]
        bid_levels = bids.shape[1]

        if ask_levels > bid_levels:
            bids = np.resize(bids, asks.shape)

        elif bid_levels > ask_levels:
            asks = np.resize(asks, bids.shape)

        depth = np.concatenate((asks, bids))

        self.emit('clear_timeout')

        self.publish(self.symbol, json.dumps(depth.tolist()))

    def clear_timeout(self):
        alog.info('### clear timeout ###')
        self.timeout = datetime.now() + timedelta(seconds=5)



@click.command()
@click.argument('symbol', type=str)
def main(**kwargs):
    emitter = DepthEmitter(**kwargs)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda: exit(0))
    signal.signal(signal.SIGTERM, lambda: exit(0))
    main()
