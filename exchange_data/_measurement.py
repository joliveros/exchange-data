import json
from datetime import datetime

import alog

from exchange_data.utils import DateTimeUtils


class Measurement(DateTimeUtils):
    def __init__(
        self,
        fields: dict = None,
        measurement: str = None,
        tags: dict = None,
        time: datetime = None
    ):
        self.fields = fields

        if isinstance(time, float):
            time = self.parse_timestamp(time)

        assert isinstance(time, datetime)
        self.time = time
        self.tags = tags
        self.measurement = measurement

    def __repr__(self):
            return alog.pformat(self.__dict__)

    def __str__(self):
        data = self.__dict__
        data['time'] = str(data['time'])
        return json.dumps(data)
