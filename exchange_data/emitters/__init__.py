import sys

from .messenger import Messenger
from .time_emitter import TimeEmitter, TimeChannels
from typing import Callable

import signal


class SignalInterceptor(object):
    def __init__(self, exit_func: Callable = None):
        if exit_func is None:
            exit_func = self.exit
        signal.signal(signal.SIGINT, exit_func)
        signal.signal(signal.SIGTERM, exit_func)

    def exit(self, *args):
        sys.exit(0)
