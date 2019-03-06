from .messenger import Messenger
from .time_emitter import TimeEmitter, TimeChannels
from typing import Callable

import signal


class SignalInterceptor(object):
    def __init__(self, exit_func: Callable):
        signal.signal(signal.SIGINT, exit_func)
        signal.signal(signal.SIGTERM, exit_func)
