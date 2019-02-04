from exchange_data import settings
from socketio import Client


class WebsocketEmitter(object):
    def __init__(self):
        self.client = Client()
        self.client.connect(f'http://{settings.WS_HOST}')

    def ws_emit(self, channel: str, data: str):
        msg = {
            'data': data,
            'channel': channel
        }
        self.client.emit('message', msg)
