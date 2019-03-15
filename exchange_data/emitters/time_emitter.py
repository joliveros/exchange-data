from datetime import datetime

import alog

from exchange_data import settings
from exchange_data.emitters.messenger import Messenger
from exchange_data.utils import NoValue
from pytimeparse.timeparse import timeparse
from time import sleep

import click


class TimeEmitter(Messenger):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.minute_counter = 0
        self.five_second_counter = 0

    @staticmethod
    def timestamp():
        return datetime.utcnow().timestamp()

    def start(self):
        while True:
            sleep(1)
            now = self.timestamp()
            self.publish(TimeChannels.Tick.value, now)

            self.minute_counter += 1
            self.five_second_counter += 1

            if self.minute_counter % 60 == 0:
                self.publish('1m', str(now))
                self.minute_counter = 0

            if self.five_second_counter % 5 == 0:
                self.five_second_counter = 0
                self.publish('5s', str(now))


    def publish(self, *args):
        alog.info(args)
        super().publish(*args)


class TimeChannels(NoValue):
    Tick = 'tick'


@click.command()
@click.argument('interval', nargs=1, required=False, default='1s')
def main(**kwargs):
    time_emitter = TimeEmitter(**kwargs)
    time_emitter.start()


if __name__ == '__main__':
    main()
