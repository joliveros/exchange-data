from bitmex_websocket import Instrument
from bitmex_websocket.constants import InstrumentChannels
from cached_property import cached_property
from datetime import datetime
from exchange_data import settings
from pyee import EventEmitter
from pytimeparse.timeparse import timeparse
from redis import Redis
from time import sleep

import alog
import asyncio
import json
import sys
import websocket


class Messenger(Redis, EventEmitter):
    def __init__(self, host: str = None):
        if host is None:
            host = settings.REDIS_HOST
        Redis.__init__(self, host)
        EventEmitter.__init__(self)

    def sub(self, channel):
        pubsub = self.pubsub()
        pubsub.subscribe([channel])
        for message in pubsub.listen():
            self.emit(channel, message)


class TimeEmitter(Messenger):
    def __init__(self, tick_interval: str = '1s'):
        super().__init__()
        self.tick_interval = timeparse(tick_interval)
        self.padding = 1100

    @property
    def next_tick(self):
        now = datetime.now()
        second_diff = (now.microsecond + self.padding)/999999
        return self.tick_interval - second_diff

    async def ticker(self):
        while True:
            sleep(self.next_tick)
            now = datetime.now()
            self.publish('tick', str(now))

    def start(self):
        loop = asyncio.get_event_loop()
        loop.create_task(self.ticker())
        loop.run_forever()


class BitmexEmitterBase(object):
    exchange = 'bitmex'

    def __init__(self, symbol: str):
        self.symbol = symbol

    @cached_property
    def channel_name(self):
        return f'{self.symbol}-{self.exchange}'


class BitmexEmitter(BitmexEmitterBase, Messenger, Instrument):
    measurements = []
    channels = [
        # InstrumentChannels.quote,
        InstrumentChannels.trade,
        InstrumentChannels.orderBookL2
    ]

    def __init__(self, symbol):
        BitmexEmitterBase.__init__(self, symbol)
        Messenger.__init__(self)
        websocket.enableTrace(settings.RUN_ENV == 'development')

        self.symbol = symbol
        Instrument.__init__(self, symbol=symbol,
                            channels=self.channels,
                            should_auth=False)

        self.on('action', self.on_action)

    def on_action(self, data):
        msg = self.channel_name, json.dumps(data)

        try:
            self.publish(*msg)
        except Exception as e:
            alog.info(e)
            sys.exit(-1)

    def start(self):
        self.run_forever()
