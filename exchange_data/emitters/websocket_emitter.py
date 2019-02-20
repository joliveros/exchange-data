from exchange_data import settings
from socketio import Client


class WebsocketEmitter(object):
    def __init__(self, websocket_emitter_enabled: bool = False, **kwargs):
        self.websocket_emitter_enabled = websocket_emitter_enabled

        self.ws_client = Client()

        if websocket_emitter_enabled:
            self.ws_client.connect(f'http://{settings.WS_HOST}')

    def ws_disconnect(self):
        self.ws_client.disconnect()

    def ws_emit(self, channel: str, data: str):
        if self.websocket_emitter_enabled:
            msg = {
                'data': data,
                'channel': channel
            }
            self.ws_client.emit('message', msg)
