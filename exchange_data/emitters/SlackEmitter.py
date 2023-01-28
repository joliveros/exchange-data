import alog
from slack_sdk import WebClient

from exchange_data import settings


class SlackEmitter(object):
    def __init__(self, channel: str, **kwargs):
        super().__init__(
            **kwargs
        )
        self.channel = channel
        self.slack_client = WebClient(token=settings.SLACK_TOKEN)

    def message(self, msg: str):
        self.slack_client.chat_postMessage(
            channel=f'#{self.channel}',
            text=msg)
