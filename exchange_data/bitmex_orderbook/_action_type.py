from enum import auto

from exchange_data.utils import NoValue


class ActionType(NoValue):
    INSERT = auto()
    UPDATE = auto()
    PARTIAL = auto()
    DELETE = auto()