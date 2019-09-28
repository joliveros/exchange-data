import logging
import sys
from abc import ABC
from enum import Enum, auto

import alog

from exchange_data import settings
from exchange_data.utils import NoValue
from pyee import EventEmitter
from redis import Redis
from typing import List

import json


class Events(NoValue):
    Message = 'message'


class MessageType(NoValue):
    subscribe = auto()
    message = auto()


class Messenger(EventEmitter, ABC):

    def __init__(self, **kwargs):
        host = settings.REDIS_HOST
        super().__init__(**kwargs)

        self.redis_client = Redis(host=host)
        self._pubsub = None
        self.on(Events.Message.value, self.handler)

    def handler(self, msg):
        if MessageType[msg['type']] == MessageType.message:
            channel_str = msg['channel'].decode()
            self.emit(channel_str, json.loads(msg['data']))

    def sub(self, channels: List):
        alog.info(channels)
        _channels = [channel.value if isinstance(channel, Enum) else channel for channel in channels]

        self._pubsub = self.redis_client.pubsub()

        self._pubsub.subscribe(_channels)

        for message in self._pubsub.listen():
            self.emit(Events.Message.value, message)

    def publish(self, channel, msg):
        if settings.LOG_LEVEL == logging.DEBUG:
            alog.debug(locals())
        self.redis_client.publish(channel, msg)

    def stop(self, *args, **kwargs):
        self._pubsub.close()
        sys.exit(0)
