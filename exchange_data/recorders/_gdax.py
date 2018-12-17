from . import Recorder
from gdax import WebsocketClient

import alog
import json


class GdaxRecorder(Recorder, WebsocketClient):
    measurements = []

    def __init__(self, symbols: list):
        Recorder.__init__(self, symbols=symbols, database_name='gdax')
        WebsocketClient.__init__(self, products=symbols, channels=['full'])

        self.start()

    def _disconnect(self):
        self.ws.send(json.dumps({
            "type": "unsubscribe",
            "channels": self.channels
        }))

        if self.ws:
                self.ws.close()

        self.on_close()

    def on_message(self, msg):
        alog.debug(msg)
        self.save_measurement('data', msg.get('product_id', None), msg)
