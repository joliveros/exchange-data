#!/usr/bin/env python


from binance.client import Client
from binance.depthcache import DepthCacheManager
from binance.websockets import BinanceSocketManager
from exchange_data.emitters import Messenger

import alog
import click
import json
import signal
import numpy as np


class DepthEmitter(Messenger):
    def __init__(self, symbol, **kwargs):
        super().__init__(**kwargs)
        self.symbol = symbol

        self.depthCache = DepthCacheManager(Client(), symbol=symbol,
                                            callback=self.message, limit=5000)

    def message(self, depthCache):
        asks = np.expand_dims(np.asarray(depthCache.get_asks()), axis=0)
        bids = np.expand_dims(np.asarray(depthCache.get_bids()), axis=0)
        ask_levels = asks.shape[1]
        bid_levels = bids.shape[1]

        if ask_levels > bid_levels:
            bids = np.resize(bids, asks.shape)

        elif bid_levels > ask_levels:
            asks = np.resize(asks, bids.shape)

        depth = np.concatenate((asks, bids))

        self.publish(self.symbol, json.dumps(depth.tolist()))


@click.command()
@click.argument('symbol', type=str)
def main(**kwargs):
    emitter = DepthEmitter(**kwargs)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda: exit(0))
    signal.signal(signal.SIGTERM, lambda: exit(0))
    main()
