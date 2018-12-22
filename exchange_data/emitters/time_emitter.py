from bitmex_websocket.constants import NoValue
from datetime import datetime
from exchange_data.emitters.messenger import Messenger
from pytimeparse.timeparse import timeparse
from time import sleep

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
        next_tick = self.tick_interval - second_diff

        if next_tick > 0:
            return next_tick
        else:
            return 0

    def start(self):
        while True:
            sleep(self.next_tick)
            now = datetime.now()
            self.publish(TimeChannels.Tick.value, str(now))


class TimeChannels(NoValue):
    Tick = 'tick'


@click.command()
@click.argument('interval', nargs=1, required=False, default='1s')
def main(interval: str):
    time_emitter = TimeEmitter(interval)
    time_emitter.start()


if __name__ == '__main__':
    main()
