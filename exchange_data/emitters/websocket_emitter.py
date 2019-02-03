import alog
from socketio import Client


class WebsocketEmitter(object):
    def __init__(self):
        self.client = Client()
        self.client.connect('http://proxy')

    def ws_emit(self, channel: str, msg: str):
        self.client.emit(channel, msg)
