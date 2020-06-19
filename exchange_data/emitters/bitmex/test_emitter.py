#!/usr/bin/env python
import asyncio
from collections import deque

from dateutil import tz

from exchange_data import settings, Database, Measurement
from exchange_data.bitmex_orderbook import BitmexOrderBook
from exchange_data.channels import BitmexChannels
from exchange_data.emitters import Messenger, TimeChannels, SignalInterceptor
from exchange_data.emitters.bitmex import BitmexEmitterBase
from exchange_data.emitters.bitmex._orderbook_l2_emitter import OrderBookL2Emitter
from exchange_data.orderbook._ordertree import OrderTree
from exchange_data.utils import NoValue, DateTimeUtils, EventEmitterBase
from functools import lru_cache
from numpy.core.multiarray import ndarray
from pyee import EventEmitter, AsyncIOEventEmitter

import alog
import click
import gc
import json
import numpy as np
import sys
import traceback



class TestEmitter(
    Messenger,
    SignalInterceptor,
    DateTimeUtils,
):
    event_emitter_class = AsyncIOEventEmitter

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(
            **kwargs
        )

        self.on(TimeChannels.Tick.value, self.test_handler)

    async def test_handler(self, data):
        alog.info(data)
        self.publish('test', '### jose was here ###')

    def start(self, channels=[]):
        self.sub([
            TimeChannels.Tick,
        ] + channels)

    def exit(self, *args):
        self.stop()
        sys.exit(0)


@click.command()
def main(**kwargs):
    loop = asyncio.get_event_loop()
    loop.set_debug(True)

    recorder = TestEmitter(
        **kwargs
    )

    try:
        loop.run_until_complete(recorder.start())
    except Exception as e:
        traceback.print_exc()
        recorder.stop()
        sys.exit(-1)


if __name__ == '__main__':
    main()
