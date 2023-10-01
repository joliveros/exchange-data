from abc import ABC
from enum import Enum, auto
from inspect import signature

from statsd import StatsClient

from exchange_data import settings
from exchange_data.utils import NoValue, EventEmitterBase
from pyee import EventEmitter
from redis import Redis
from typing import List

import alog
import json
import logging
import sys


class Events(NoValue):
    Message = 'message'


class MessageType(NoValue):
    subscribe = auto()
    message = auto()


class Messenger(EventEmitterBase, StatsClient):

    def __init__(self, decode=True, stats_prefix=None, **kwargs):
        host = settings.REDIS_HOST
        super().__init__(**kwargs)

        StatsClient.__init__(self, host='telegraf', prefix=stats_prefix)

        self.decode = decode

        redis_params = [param for param in list(signature(Redis).parameters)]
        redis_kwargs = {param: kwargs[param] for param in redis_params
                        if param in kwargs}

        self.redis_client = Redis(host=host, **redis_kwargs)
        self._pubsub = None

        if 'channels' in vars(self):
            self.channels += []
        else:
            self.channels = []

        self.on(Events.Message.value, self.handler)

    def _send(self, data):
        """Send data to statsd."""
        self._sock.sendto(data.encode('ascii'), self._addr)

    def handler(self, msg):
        if MessageType[msg['type']] == MessageType.message:
            channel_str = msg['channel'].decode()
            if self.decode:
                self.emit(channel_str, json.loads(msg['data']))
            else:
                self.emit(channel_str, msg['data'])

    def sub(self, channels: List):
        _channels = [
            channel.value if isinstance(channel, Enum) else channel
            for channel in channels
        ]

        alog.info(_channels)

        self._pubsub = self.redis_client.pubsub()

        self._pubsub.subscribe(_channels)

        for message in self._pubsub.listen():
            self.emit(Events.Message.value, message)

    def publish(self, channel, msg):
        if settings.LOG_LEVEL == logging.DEBUG:
            alog.debug(alog.pformat(locals()))
        self.redis_client.publish(channel, msg)

    def stop(self, *args, **kwargs):
        self._pubsub.close()
        sys.exit(0)

    def gauge(self, *args, **kwargs):
        super().gauge(*args, **kwargs)

    def incr(self, *args, **kwargs):
        super().incr(*args, **kwargs)
