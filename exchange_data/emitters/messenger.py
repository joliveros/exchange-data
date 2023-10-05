import asyncio
from abc import ABC
from datetime import datetime
import time
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
    Message = "message"


class MessageType(NoValue):
    subscribe = auto()
    message = auto()


class Messenger(EventEmitterBase, StatsClient):
    def __init__(self, max_empty_msg_count=5, decode=True, stats_prefix=None, **kwargs):
        self.max_empty_msg_count = max_empty_msg_count
        self.last_channel_msg = dict()
        host = settings.REDIS_HOST
        super().__init__(**kwargs)

        StatsClient.__init__(self, host="telegraf", prefix=stats_prefix)

        self.empty_msg_count = 0
        self.decode = decode

        redis_params = [param for param in list(signature(Redis).parameters)]
        redis_kwargs = {
            param: kwargs[param] for param in redis_params if param in kwargs
        }

        self.redis_client = Redis(host=host, **redis_kwargs)
        self._pubsub = None

        if "channels" in vars(self):
            self.channels += []
        else:
            self.channels = []

        for channel in self.channels:
            self.last_channel_msg[channel] = datetime.now()

        self.on(Events.Message.value, self.handler)

    def _send(self, data):
        """Send data to statsd."""
        self._sock.sendto(data.encode("ascii"), self._addr)

    def handler(self, msg):
        if MessageType[msg["type"]] == MessageType.message:
            channel_str = msg["channel"].decode()
            if self.decode:
                self.emit(channel_str, json.loads(msg["data"]))
            else:
                self.emit(channel_str, msg["data"])

    def sub(self, channels: List):
        _channels = [
            channel.value if isinstance(channel, Enum) else channel
            for channel in channels
        ]

        alog.info(_channels)

        for channel in _channels:
            self.last_channel_msg[channel] = datetime.now()

        self._pubsub = self.redis_client.pubsub()

        self._pubsub.subscribe(_channels)

        for message in self._pubsub.listen():
            self.emit(Events.Message.value, message)

    def publish(self, channel, msg):
        if settings.LOG_LEVEL == logging.DEBUG:
            alog.debug(alog.pformat(locals()))

        if channel not in self.last_channel_msg:
            self.last_channel_msg[channel] = datetime.now()

        since_last_msg = datetime.now() - self.last_channel_msg[channel]

        self.timing(f"{channel}_last_message", since_last_msg)
        self.redis_client.publish(channel, msg)
        self.last_channel_msg[channel] = datetime.now()

    def stop(self, *args, **kwargs):
        self._pubsub.close()
        sys.exit(0)

    def gauge(self, *args, **kwargs):
        super().gauge(*args, **kwargs)

    def incr(self, *args, **kwargs):
        super().incr(*args, **kwargs)

    def reset_empty_msg_count(self):
        self.empty_msg_count = 0

    def increase_empty_msg_count(self):
        if self.empty_msg_count > self.max_empty_msg_count:
            alog.info("### exiting due to excess lag ##")
            self.stream_is_crashing(self.stream_id)

        self.empty_msg_count += 1

    def stream_is_crashing(self, stream_id, error_msg=False):
        alog.debug(f"## restart stream {stream_id} ##")
        self.set_restart_request(stream_id)

