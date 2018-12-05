import asyncio
import json
import sys
from datetime import datetime
from time import sleep

import alog
import websocket
from bitmex_websocket import Instrument
from bitmex_websocket.constants import InstrumentChannels
from pytimeparse.timeparse import timeparse
from redis import Redis

from exchange_data import settings


class MessageEmitter(Redis):
    def __init__(self):
        super().__init__(settings.REDIS_HOST)


class TimeEmitter(MessageEmitter):
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
            alog.info(f'#### {now} ####')
            self.publish('tick', str(now))

    def start(self):
        loop = asyncio.get_event_loop()
        loop.create_task(self.ticker())
        loop.run_forever()


class BitmexEmitter(MessageEmitter, Instrument):
    measurements = []
    channels = [
        InstrumentChannels.quote,
        InstrumentChannels.trade,
        InstrumentChannels.orderBookL2
    ]

    def __init__(self, symbol):
        MessageEmitter.__init__(self)
        websocket.enableTrace(settings.RUN_ENV == 'development')
        self.exchange = 'bitmex'

        self.symbol = symbol
        Instrument.__init__(self, symbol=symbol,
                            channels=self.channels,
                            should_auth=False)

        self.on('action', self.on_action)

    def on_action(self, data):
        msg = f'{self.symbol}{self.exchange}', json.dumps(data)

        try:
            self.publish(*msg)
        except Exception as e:
            alog.info(e)
            sys.exit(-1)

    def start(self):
        self.run_forever()
