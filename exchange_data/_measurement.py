from datetime import datetime
from dateutil import parser
from pandas import DataFrame

from exchange_data.utils import DateTimeUtils

import alog
import json


class Measurement(DateTimeUtils):
    def __init__(
        self,
        fields: dict or DataFrame,
        measurement: str,
        time: datetime = None,
        tags: dict = []
    ):
        if type(fields) == DataFrame:
            fields = dict(data=fields.to_json())

        self.fields = fields

        if time is None:
            time = self.now()

        if isinstance(time, float):
            time = self.parse_timestamp(time)
        elif isinstance(time, str):
            time = self.parse_datetime_str(time)

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
