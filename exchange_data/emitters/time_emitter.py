from datetime import datetime
from exchange_data import settings
from exchange_data.emitters.messenger import Messenger
from exchange_data.utils import NoValue
from pytimeparse.timeparse import timeparse
from time import sleep

import click


class TimeEmitter(Messenger):

    def __init__(self, tick_interval: str = settings.TICK_INTERVAL):
        super().__init__()
        self.tick_interval = timeparse(tick_interval)
        self.padding = 1100
        self.minute_counter = 0

    @staticmethod
    def timestamp():
        return datetime.utcnow().timestamp()

    def start(self):
        while True:
            sleep(1)
            now = self.timestamp()

            self.minute_counter += 1
            if self.minute_counter % 60 == 0:
                self.publish('1m', str(now))
                self.minute_counter = 0
            timestamp = self.timestamp()

            self.publish(TimeChannels.Tick.value, timestamp)


class TimeChannels(NoValue):
    Tick = 'tick'


@click.command()
@click.argument('interval', nargs=1, required=False, default='1s')
def main(interval: str):
    time_emitter = TimeEmitter(interval)
    time_emitter.start()


if __name__ == '__main__':
    main()
