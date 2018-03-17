from . import Recorder
from gdax import WebsocketClient
import alog


class GdaxRecorder(Recorder, WebsocketClient):
    measurements = []

    def __init__(self, symbols: list):
        Recorder.__init__(self, symbols=symbols, database_name='gdax')
        WebsocketClient.__init__(self, products=symbols, channels=['level2'])

        self.start()

    def on_message(self, msg):
        alog.debug(msg)
        self.save_measurement('level2', msg.get('product_id', None), msg)
