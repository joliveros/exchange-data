import json
import alog

from pyee import EventEmitter
from abc import ABC
from . import settings
from ._buffer import Buffer
from ._database import Database
from ._measurement import Measurement

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


__all__ = [
    Buffer,
    NumpyEncoder
]


class IEventEmitter(ABC):
    def __init__(self, **kwargs):
        super(ABC, self).__init__()

    def on(self, event: str, callback):
        raise NotImplementedError()


class EventEmitterBase(ABC, EventEmitter):
    def __init__(self, **kwargs):
        EventEmitter.__init__(self)
        super().__init__(**kwargs)

    def on(self, event, callback):
        EventEmitter.on(self, event, callback)
