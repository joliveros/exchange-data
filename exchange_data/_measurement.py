import alog


class Measurement(object):
    def __init__(
        self,
        fields: dict = None,
        measurement: str = None,
        tags: dict = None,
        timestamp: float = None
    ):
        self.fields = fields
        self.timestamp = timestamp
        self.tags = tags
        self.measurement = measurement

    def __repr__(self):
            return alog.pformat(self.__dict__)
