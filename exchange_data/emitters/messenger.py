from enum import Enum, auto
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


class Messenger(Redis, EventEmitter):

    def __init__(self):
        self.host: str = settings.REDIS_HOST

        Redis.__init__(self, self.host)
        EventEmitter.__init__(self)
        self._pubsub = None
        self.on(Events.Message.value, self.handler)

    def handler(self, msg):
        if MessageType[msg['type']] == MessageType.message:
            channel_str = msg['channel'].decode()
            self.emit(channel_str, json.loads(msg['data']))

    def sub(self, channels: List):
        _channels = [channel.value if isinstance(channel, Enum) else channel for channel in channels]

        self._pubsub = self.pubsub()

        self._pubsub.subscribe(_channels)

        for message in self._pubsub.listen():
            self.emit(Events.Message.value, message)

