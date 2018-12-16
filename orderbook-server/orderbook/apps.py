import alog as alog
from django.apps import AppConfig


class OrderbookConfig(AppConfig):
    name = 'orderbook'

    def ready(self):
        alog.debug(f'### START {self.__class__.name} ###')
