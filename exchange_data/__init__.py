import json

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


