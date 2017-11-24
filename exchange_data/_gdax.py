from . import Recorder
from . import settings
from gdax import WebsocketClient
import alog

alog.set_level(settings.LOG_LEVEL)


class GdaxRecorder(Recorder, WebsocketClient):
    measurements = []
    channels = [
        'trade',
        'quote',
        'orderBookL2'
    ]

    def __init__(self, symbols):
        Recorder.__init__(symbols, database_name='gdax')
        WebsocketClient.__init__(self)

        self.products = symbols

    def on_open(self):
        self.url = "wss://ws-feed.gdax.com/"

    def on_message(self, msg):
        alog.debug(msg)
