from . import Recorder
from dateutil import parser
from exchange_data.utils import DateTimeUtils
from gdax import WebsocketClient

import json


class GdaxRecorder(Recorder, WebsocketClient, DateTimeUtils):
    measurements = []

    def __init__(self, symbols: list):
        DateTimeUtils.__init__(self)
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
        if 'time' in msg:
            msg['time'] = parser.parse(msg.get('time'))
        else:
            msg['time'] = self.now()

        self.save_measurement('data', msg.get('product_id', None), msg)
