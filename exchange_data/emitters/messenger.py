from enum import Enum
from exchange_data import settings
from exchange_data.utils import NoValue
from pyee import EventEmitter
from redis import Redis
from typing import Callable, List


class Events(NoValue):
    Message = 'message'


class Messenger(Redis, EventEmitter):

    def __init__(self, handler: Callable = None, host: str = None):
        self.handler = handler
        if host is None:
            host = settings.REDIS_HOST

        self.host: str = host

        Redis.__init__(self, host)
        EventEmitter.__init__(self)

        if self.handler:
            self.on(Events.Message.value, self.handler)

    def sub(self, channels: List[Enum]):
        _channels = [channel.value for channel in channels]

        pubsub = self.pubsub()

        pubsub.subscribe(_channels)

        for message in pubsub.listen():
            self.emit(Events.Message.value, message)

