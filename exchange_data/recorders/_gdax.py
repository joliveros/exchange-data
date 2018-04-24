from . import Recorder
from gdax import WebsocketClient
import alog


class GdaxRecorder(Recorder, WebsocketClient):
    measurements = []

    def __init__(self, symbols: list):
        Recorder.__init__(self, symbols=symbols, database_name='gdax')
        WebsocketClient.__init__(self, products=symbols, channels=['full'])

        self.start()

    def on_message(self, msg):
        alog.debug(msg)
        self.save_measurement('data', msg.get('product_id', None), msg)
