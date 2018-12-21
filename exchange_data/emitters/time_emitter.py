from bitmex_websocket.constants import NoValue
from datetime import datetime
from exchange_data.emitters.messenger import Messenger
from pytimeparse.timeparse import timeparse
from time import sleep

import asyncio
import click


class TimeEmitter(Messenger):

    def __init__(self, tick_interval: str = '1s'):
        super().__init__()
        self.tick_interval = timeparse(tick_interval)
        self.padding = 1100

    @property
    def next_tick(self):
        now = datetime.now()
        second_diff = (now.microsecond + self.padding)/999999
        return self.tick_interval - second_diff

    async def ticker(self):
        while True:
            sleep(self.next_tick)
            now = datetime.now()
            self.publish(TimeChannels.Tick.value, str(now))

    def start(self):
        loop = asyncio.get_event_loop()
        loop.create_task(self.ticker())
        loop.run_forever()


class TimeChannels(NoValue):
    Tick = 'tick'


@click.command()
@click.argument('interval', nargs=1, required=False, default='1s')
def main(interval: str):
    time_emitter = TimeEmitter(interval)
    time_emitter.start()


if __name__ == '__main__':
    main()
