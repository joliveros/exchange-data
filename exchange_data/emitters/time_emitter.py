#!/usr/bin/env python

from datetime import datetime
from exchange_data import settings
from exchange_data.emitters.messenger import Messenger
from exchange_data.utils import NoValue, DateTimeUtils, EventEmitterBase
from time import sleep
import alog
import click

alog.set_level(settings.LOG_LEVEL)


class TimeEmitter(Messenger, DateTimeUtils):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.minute_counter = 0
        self.four_minute_counter = 0
        self.five_minute_counter = 0
        self.three_minute_counter = 0
        self.two_minute_counter = 0
        self.two_second_counter = 0
        self.fifteen_second_counter = 0
        self.twenty_second_counter = 0
        self.five_second_counter = 0
        self.should_stop = False
        self.last_dt = self.now().replace(microsecond=0)

    @staticmethod
    def timestamp():
        return datetime.utcnow().timestamp()

    def timestamp_str(self):
        return str(self.last_dt.timestamp())

    def start(self):
        while not self.should_stop:
            sleep(1/10**9)

            if self.tick():
                sleep(99/100)
                t = self.timestamp_str()
                self.publish(TimeChannels.Tick.value, t)

                self.five_second_counter += 1
                self.four_minute_counter += 1
                self.five_minute_counter += 1
                self.minute_counter += 1
                self.three_minute_counter += 1
                self.two_minute_counter += 1
                self.two_second_counter += 1
                self.fifteen_second_counter += 1
                self.twenty_second_counter += 1

                if self.two_second_counter % 2 == 0:
                    self.two_second_counter = 0
                    self.publish('2s', t)

                if self.fifteen_second_counter % 15 == 0:
                    self.fifteen_second_counter = 0
                    self.publish('15s', t)

                if self.twenty_second_counter % 20 == 0:
                    self.twenty_second_counter = 0
                    self.publish('20s', t)

                if self.minute_counter % 60 == 0:
                    self.publish('1m', t)
                    self.minute_counter = 0

                if self.two_minute_counter % (60 * 2) == 0:
                    self.publish('2m', t)
                    self.three_minute_counter = 0

                if self.three_minute_counter % (60 * 3) == 0:
                    self.publish('3m', t)
                    self.three_minute_counter = 0

                if self.four_minute_counter % 240 == 0:
                    self.publish('4m', t)
                    self.four_minute_counter = 0

                if self.five_minute_counter % 300 == 0:
                    self.publish('5m', t)
                    self.five_minute_counter = 0

                if self.minute_counter % 30 == 0:
                    self.publish('30s', t)

                if self.minute_counter % 10 == 0:
                    self.publish('10s', t)

                if self.five_second_counter % 5 == 0:
                    self.five_second_counter = 0
                    self.publish('5s', t)

    def publish(self, *args):
        alog.debug(args)
        self.redis_client.publish(*args)

    def stop(self, *args, **kwargs):
        super().stop(*args, **kwargs)

    def tick(self):
        dt = self.now()
        diff = dt - self.last_dt

        if diff.total_seconds() >= 1.0:
            self.last_dt = dt.replace(microsecond=0)
            return self.last_dt


class TimeChannels(NoValue):
    Tick = 'tick'


@click.command()
def main(**kwargs):
    time_emitter = TimeEmitter(**kwargs)
    time_emitter.start()


if __name__ == '__main__':
    main()
