from datetime import datetime

import alog


class Measurement(object):
    def __init__(
        self,
        fields: dict = None,
        measurement: str = None,
        tags: dict = None,
        time: datetime = None
    ):
        self.fields = fields
        assert isinstance(time, datetime)
        self.time = time
        self.tags = tags
        self.measurement = measurement

    def __repr__(self):
            return alog.pformat(self.__dict__)
